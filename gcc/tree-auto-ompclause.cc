#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "rtl.h"
#include "tree.h"
#include "gimple.h"
#include "cfghooks.h"
#include "tree-pass.h"
#include "ssa.h"
#include "cgraph.h"
#include "diagnostic-core.h"
#include "fold-const.h"
#include "calls.h"
#include "except.h"
#include "cfganal.h"
#include "cfgcleanup.h"
#include "gimple-iterator.h"
#include "tree-cfg.h"
#include "tree-into-ssa.h"
#include "tree-ssa.h"
#include "tree-inline.h"
#include "langhooks.h"
#include "cfgloop.h"
#include "gimple-low.h"
#include "stringpool.h"
#include "attribs.h"
#include "asan.h"
#include "gimplify.h"
#include "tree-iterator.h"

#include "../libcpp/omp_global.h"
#include <string>
#include <sstream>
#include <vector>
#include <map>
using namespace std;
// tree globals = lang_hooks.decls.getdecls();
bool outer_omp_func = false;
bool omp_for = false;

// 存储  函数名-子句列表的字典
map<string, vector<string>> fn_name_args;

std::vector<std::string> splitString(const std::string &input, char delimiter)
{
    std::vector<std::string> tokens;
    std::istringstream iss(input);
    std::string token;
    while (std::getline(iss, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}
void get_fnname_args(string str)
{

    map<string, vector<string>> myDictionary;

    // 获取函数名
    string s = str.substr(0, str.find("("));
    // 从函数名开始截断  (linear(1:1),simdlen(8))
    str = str.substr(str.find("("));
    // cout << str << endl;
    // 删除两端的括号
    str = str.substr(str.find("(") + 1, str.rfind(")") - 1);
    // cout << str << endl;

    // 获取参数列表
    vector<string> args = splitString(str, ',');

    fn_name_args.insert({s, args});
}
void process_omp_clause()
{
    // 创建一个空的字符串
    string str;
    str = extern_omp_clause;
    string newstr;

    // 删除子句的字符串
    for (char s : str)
    {
        if (s != ' ')
            newstr += s;
    }
    str = newstr;

    // 去除前后的大括号
    str = str.replace(0, 1, "");
    str = str.replace(str.length() - 1, str.length(), "");

    // 根据;分割，获取每个函数的名字以及参数列表
    vector<string> tokens = splitString(str, ';');

    for (const auto &token : tokens)
    {
        get_fnname_args(token);
    }
}

// 检查并添加 noinline 属性 !DECL_IN_SYSTEM_HEADER (called_function_decl)
static void check_and_add_noinline_attribute(tree function_decl)
{
    //  检查并添加 omp declare simd 属性
    if (function_decl)
    {
        tree called_function_decl = TREE_OPERAND(function_decl, 0);
        if (called_function_decl && !DECL_IN_SYSTEM_HEADER(called_function_decl))
        {
            if (!lookup_attribute("omp declare simd", DECL_ATTRIBUTES(called_function_decl)))
            {
                tree omp_declare_simd_attr = get_identifier("omp declare simd");
                DECL_ATTRIBUTES(called_function_decl) = tree_cons(omp_declare_simd_attr, NULL_TREE, DECL_ATTRIBUTES(called_function_decl));
            }
            // 检查是否已具有 noinline 属性
            if (!lookup_attribute("noinline", DECL_ATTRIBUTES(called_function_decl)))
            {
                tree noinline_attr = get_identifier("noinline");
                DECL_ATTRIBUTES(called_function_decl) = tree_cons(noinline_attr, NULL_TREE, DECL_ATTRIBUTES(called_function_decl));
            }
        }
    }
}

/* 遍历函数体或循环中的语句 */
static void traverse_function_body(tree stmt)
{

    // for (int i = 0; i < TREE_OPERAND_LENGTH(stmt_sum); i++)
    // {
    //   tree stmt = TREE_OPERAND(stmt_sum, i);
    if (stmt && TREE_CODE(stmt) == BIND_EXPR)
    {
        // 获取 BIND_EXPR 中的语句列表
        tree body = BIND_EXPR_BODY(stmt);
        traverse_function_body(body);
    }
    else if (stmt && TREE_CODE(stmt) == OMP_SIMD)
    {
        // check_loop_omp(stmt);
        omp_for = true;
        traverse_function_body(TREE_OPERAND(stmt, 0));
        omp_for = false;
    }
    else if (stmt && TREE_CODE(stmt) == STATEMENT_LIST)
    {
        // 对语句列表进行遍历
        tree_stmt_iterator iter;
        for (iter = tsi_start(stmt); !tsi_end_p(iter); tsi_next(&iter))
        {

            traverse_function_body(tsi_stmt(iter));
        }
    }
    else if ((outer_omp_func || omp_for) && stmt && TREE_CODE(stmt) == CALL_EXPR)
    {

        if (CALL_EXPR_FN(stmt) && TREE_OPERAND_LENGTH(CALL_EXPR_FN(stmt)) > 0)
        {
            if (TREE_OPERAND(CALL_EXPR_FN(stmt), 0) && !DECL_IN_SYSTEM_HEADER(TREE_OPERAND(CALL_EXPR_FN(stmt), 0)) && TREE_CODE(TREE_OPERAND(CALL_EXPR_FN(stmt), 0)) == FUNCTION_DECL)
                check_and_add_noinline_attribute(CALL_EXPR_FN(stmt));
            // 递归处理其他表达式中的函数调用

            for (int j = 1; j < TREE_OPERAND_LENGTH(CALL_EXPR_FN(stmt)); j++)
            {
                traverse_function_body(TREE_OPERAND(CALL_EXPR_FN(stmt), j));
            }
        }
    }
    else
    {
        if (stmt && TREE_OPERAND_LENGTH(stmt) > 0)
        {
            for (int j = 0; j < TREE_OPERAND_LENGTH(stmt); j++)
            {
                traverse_function_body(TREE_OPERAND(stmt, j));
            }
        }
    }
    // }
}

tree process_sub_clause(string clause, tree clauselist)
{
    string clause_name = clause.substr(0, clause.find("("));
    string clause_arg = clause.substr(clause.find("(") + 1, clause.find(")") - clause.find("(") - 1);
    if (clause_name == "linear")
    {
        vector<string> args = splitString(clause_arg, ':');
        int num1, num2;
        num1 = std::stoi(args[0]);
        num2 = std::stoi(args[1]);

        tree linear_attr = (tree)ggc_internal_alloc((sizeof(struct tree_omp_clause) + (3 - 1) * sizeof(tree)));
        tree linear_op1 = build_int_cst(NULL_TREE, num1); // 第一个参数为 2
        tree linear_op2 = build_int_cst(NULL_TREE, num2); // 第二个参数为 3

        TREE_SET_CODE(linear_attr, OMP_CLAUSE);
        OMP_CLAUSE_SET_CODE(linear_attr, OMP_CLAUSE_LINEAR);
        OMP_CLAUSE_DECL(linear_attr) = linear_op1;
        OMP_CLAUSE_LINEAR_STEP(linear_attr) = linear_op2;
        OMP_CLAUSE_CHAIN(linear_attr) = clauselist;
        // 添加到 clauselist 链表
        // clauselist = tree_cons(NULL_TREE, linear_attr, clauselist);
        return linear_attr;
        // clauselist = linear_attr;
    }
    else if (clause_name == "aligned")
    {
        vector<string> args = splitString(clause_arg, ':');
        int num1, num2;
        num1 = std::stoi(args[0]);
        num2 = std::stoi(args[1]);

        tree aligned_attr = (tree)ggc_internal_alloc((sizeof(struct tree_omp_clause) + (2 - 1) * sizeof(tree)));
        tree aligned_op1 = build_int_cst(NULL_TREE, num1); // 第一个参数为 2
        tree aligned_op2 = build_int_cst(NULL_TREE, num2); // 第二个参数为 3

        TREE_SET_CODE(aligned_attr, OMP_CLAUSE);
        OMP_CLAUSE_SET_CODE(aligned_attr, OMP_CLAUSE_ALIGNED);
        OMP_CLAUSE_DECL(aligned_attr) = aligned_op1;
        OMP_CLAUSE_ALIGNED_ALIGNMENT(aligned_attr) = aligned_op2;
        OMP_CLAUSE_CHAIN(aligned_attr) = clauselist;
        // 添加到 clauselist 链表
        // clauselist = tree_cons(NULL_TREE, linear_attr, clauselist);
        return aligned_attr;
    }
    else if (clause_name == "simdlen")
    {
        int num = std::stoi(clause_arg);
        tree simdlen_attr = (tree)ggc_internal_alloc((sizeof(struct tree_omp_clause) + (1 - 1) * sizeof(tree)));
        // const char * name = clause_name;
        tree simdlen_value = build_int_cst(NULL_TREE, num); // 8 是安全长度的值

        TREE_SET_CODE(simdlen_attr, OMP_CLAUSE);
        OMP_CLAUSE_SET_CODE(simdlen_attr, OMP_CLAUSE_SIMDLEN);
        OMP_CLAUSE_SIMDLEN_EXPR(simdlen_attr) = simdlen_value;

        // 添加到 linear_attr 链表
        OMP_CLAUSE_CHAIN(simdlen_attr) = clauselist;
        return simdlen_attr;
    }
    else if (clause_name == "uniform")
    {
        int num = std::stoi(clause_arg);
        tree uniform_attr = (tree)ggc_internal_alloc((sizeof(struct tree_omp_clause) + (1 - 1) * sizeof(tree)));
        // const char * name = clause_name;
        tree uniform_value = build_int_cst(NULL_TREE, num); // 8 是安全长度的值  

        TREE_SET_CODE(uniform_attr, OMP_CLAUSE);
        OMP_CLAUSE_SET_CODE(uniform_attr, OMP_CLAUSE_UNIFORM);
        OMP_CLAUSE_DECL(uniform_attr) = uniform_value;

        // 添加到 linear_attr 链表
        OMP_CLAUSE_CHAIN(uniform_attr) = clauselist;
        return uniform_attr;
    }
    else if (clause_name == "inbranch" || clause_name == "notinbranch")
    {
       
        tree branch_attr = (tree)ggc_internal_alloc((sizeof(struct tree_omp_clause) + (0 - 1) * sizeof(tree)));
      
        TREE_SET_CODE(branch_attr, OMP_CLAUSE);
        if (clause_name == "inbranch")
            OMP_CLAUSE_SET_CODE(branch_attr, OMP_CLAUSE_INBRANCH);
        else
            OMP_CLAUSE_SET_CODE(branch_attr, OMP_CLAUSE_NOTINBRANCH);


        // 添加到 linear_attr 链表
        OMP_CLAUSE_CHAIN(branch_attr) = clauselist;
        return branch_attr;
    }
    return clauselist;
}

/* 遍历函数的回调函数 */
static void my_callback_function(cgraph_node *node)
{
    tree decl = FUNCTION_DECL_CHECK(node->decl); // 获取函数声明节点
    const char *func_name = IDENTIFIER_POINTER(DECL_NAME(decl));
    if (decl && TREE_CODE(decl) == FUNCTION_DECL)
    {
        auto it = fn_name_args.find(func_name);
        if (it != fn_name_args.end())
        {
            tree clauselist = NULL;
            for (const string &clause : it->second)
            {
                if (clause.length() > 0)
                    clauselist = process_sub_clause(clause, clauselist);
            }
            clauselist = tree_cons(NULL_TREE, clauselist, NULL_TREE);

            // 构建 omp declare simd 属性链表
            tree declare_simd_attr = build_tree_list(get_identifier("omp declare simd"), clauselist);
            // 将新的属性链表连接到函数声明的属性链表上
            DECL_ATTRIBUTES(decl) = declare_simd_attr;
            // check_and_add_noinline_attribute(decl);
                        // 检查是否已具有 noinline 属性
            if (!lookup_attribute("noinline", DECL_ATTRIBUTES(decl)))
            {
                tree noinline_attr = get_identifier("noinline");
                DECL_ATTRIBUTES(decl) = tree_cons(noinline_attr, NULL_TREE, DECL_ATTRIBUTES(decl));
            }
        }
        // 获取函数体
        tree body = DECL_SAVED_TREE(decl);
        // 检查是否具有 omp declare simd 属性
        if (lookup_attribute("omp declare simd", DECL_ATTRIBUTES(decl)))
        {
            outer_omp_func = true;
        }
        // 对函数体进行遍历
        traverse_function_body(body);
        outer_omp_func = false;
    }
}
/* 遍历函数声明 */
static void iterate_functions(void)
{
    tree cccc = build_int_cst(NULL_TREE, 8); // 8 是安全长度的值  
    struct cgraph_node *node;
    if (extern_omp_clause[0] != '\0')
        process_omp_clause();
    FOR_EACH_DEFINED_FUNCTION(node)
    {
        my_callback_function(node);
    }
}

namespace
{

    const pass_data pass_data_auto_ompclause =
        {
            GIMPLE_PASS,       /* type */
            "auto_ompclause",  /* name */
            OPTGROUP_NONE,     /* optinfo_flags */
            TV_AUTO_OMPCLAUSE, /* tv_id */
            0,                 /* properties_required */
            0,                 /* properties_provided */
            0,                 /* properties_destroyed */
            0,                 /* todo_flags_start */
            0,                 /* todo_flags_finish */
    };

    class pass_auto_ompclause : public gimple_opt_pass
    {
    private:
        bool has_traversed_functions; // 用于标记是否已经遍历过函数
    public:
        pass_auto_ompclause(gcc::context *ctxt)
            : gimple_opt_pass(pass_data_auto_ompclause, ctxt)
        {
        }

        virtual bool gate(function *fun)
        {
            return true;
        }

        virtual unsigned int execute(function *);

    }; // class pass_cleanup_eh

    unsigned int
    pass_auto_ompclause::execute(function *fun)
    {
        // 只执行一次遍历函数的操作
        if (!has_traversed_functions)
        {
            iterate_functions();
            has_traversed_functions = true;
            // printf("ONETIMEHELLO !!\n");
        }
        // my_callback_function();
        // printf("HELLO !!\n");
        return 1;
    }

} // anon namespace

gimple_opt_pass *
make_pass_auto_ompclause(gcc::context *ctxt)
{
    return new pass_auto_ompclause(ctxt);
}