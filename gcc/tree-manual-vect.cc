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

#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "target.h"
#include "tree.h"
#include "gimple.h"
#include "cfghooks.h"
#include "alloc-pool.h"
#include "tree-pass.h"
#include "ssa.h"
#include "cgraph.h"
#include "pretty-print.h"
#include "diagnostic-core.h"
#include "fold-const.h"
#include "stor-layout.h"
#include "cfganal.h"
#include "gimplify.h"
#include "gimple-iterator.h"
#include "gimplify-me.h"
#include "gimple-walk.h"
#include "langhooks.h"
#include "tree-cfg.h"
#include "tree-into-ssa.h"
#include "tree-dfa.h"
#include "cfgloop.h"
#include "symbol-summary.h"
#include "ipa-param-manipulation.h"
#include "tree-eh.h"
#include "varasm.h"
#include "stringpool.h"
#include "attribs.h"
#include "omp-simd-clone.h"
#include "omp-simd-clone.cc"

// static struct cgraph_simd_clone * simd_clone_struct_alloc (int nargs)
// {
//   struct cgraph_simd_clone *clone_info;
//   size_t len = (sizeof (struct cgraph_simd_clone)
// 		+ nargs * sizeof (struct cgraph_simd_clone_arg));
//   clone_info = (struct cgraph_simd_clone *)
// 	       ggc_internal_cleared_alloc (len);
//   return clone_info;
// }

static tree
simd_clone_manual_mangle(struct cgraph_node *node,
                         struct cgraph_simd_clone *clone_info)
{
  char vecsize_mangle = clone_info->vecsize_mangle;
  char mask = clone_info->inbranch ? 'M' : 'N';
  poly_uint64 simdlen = clone_info->simdlen;
  unsigned int n;
  pretty_printer pp;

  gcc_assert(vecsize_mangle && maybe_ne(simdlen, 0U));

  pp_string(&pp, "_ZGV");
  pp_character(&pp, vecsize_mangle);
  pp_character(&pp, mask);
  /* For now, simdlen is always constant, while variable simdlen pp 'n'.  */
  unsigned int len = simdlen.to_constant();
  pp_decimal_int(&pp, (len));

  for (n = 0; n < clone_info->nargs; ++n)
  {
    struct cgraph_simd_clone_arg arg = clone_info->args[n];

    switch (arg.arg_type)
    {
    case SIMD_CLONE_ARG_TYPE_UNIFORM:
      pp_character(&pp, 'p');
      break;
    case SIMD_CLONE_ARG_TYPE_LINEAR_CONSTANT_STEP:
      pp_character(&pp, 'l');
      goto mangle_linear;
    case SIMD_CLONE_ARG_TYPE_LINEAR_REF_CONSTANT_STEP:
      pp_character(&pp, 'R');
      goto mangle_linear;
    case SIMD_CLONE_ARG_TYPE_LINEAR_VAL_CONSTANT_STEP:
      pp_character(&pp, 'L');
      goto mangle_linear;
    case SIMD_CLONE_ARG_TYPE_LINEAR_UVAL_CONSTANT_STEP:
      pp_character(&pp, 'U');
      goto mangle_linear;
    mangle_linear:
      gcc_assert(arg.linear_step != 0);
      if (arg.linear_step > 1)
        pp_unsigned_wide_integer(&pp, arg.linear_step);
      else if (arg.linear_step < 0)
      {
        pp_character(&pp, 'n');
        pp_unsigned_wide_integer(&pp, (-(unsigned HOST_WIDE_INT)
                                            arg.linear_step));
      }
      break;
    case SIMD_CLONE_ARG_TYPE_LINEAR_VARIABLE_STEP:
      pp_string(&pp, "ls");
      pp_unsigned_wide_integer(&pp, arg.linear_step);
      break;
    case SIMD_CLONE_ARG_TYPE_LINEAR_REF_VARIABLE_STEP:
      pp_string(&pp, "Rs");
      pp_unsigned_wide_integer(&pp, arg.linear_step);
      break;
    case SIMD_CLONE_ARG_TYPE_LINEAR_VAL_VARIABLE_STEP:
      pp_string(&pp, "Ls");
      pp_unsigned_wide_integer(&pp, arg.linear_step);
      break;
    case SIMD_CLONE_ARG_TYPE_LINEAR_UVAL_VARIABLE_STEP:
      pp_string(&pp, "Us");
      pp_unsigned_wide_integer(&pp, arg.linear_step);
      break;
    default:
      pp_character(&pp, 'v');
    }
    if (arg.alignment)
    {
      pp_character(&pp, 'a');
      pp_decimal_int(&pp, arg.alignment);
    }
  }

  pp_underscore(&pp);
  const char *str = IDENTIFIER_POINTER(DECL_ASSEMBLER_NAME(node->decl));
  if (*str == '*')
    ++str;
  pp_string(&pp, str);
  str = pp_formatted_text(&pp);

  /* If there already is a SIMD clone with the same mangled name, don't
     add another one.  This can happen e.g. for
     #pragma omp declare simd
     #pragma omp declare simd simdlen(8)
     int foo (int, int);
     if the simdlen is assumed to be 8 for the first one, etc.  */
  // for (struct cgraph_node *clone = node->simd_clones; clone;
  //      clone = clone->simdclone->next_clone)
  //   if (id_equal (DECL_ASSEMBLER_NAME (clone->decl), str))
  //     return NULL_TREE;

  return get_identifier(str);
}

static struct cgraph_node *
manual_declare_create(struct cgraph_node *old_node)
{
  struct cgraph_node *new_node;

  tree old_decl = old_node->decl;
  tree new_decl = copy_node(old_node->decl);
  tree new_func_name = get_identifier("VFOOFOO");
  DECL_NAME(new_decl) = new_func_name;
  SET_DECL_ASSEMBLER_NAME(new_decl, DECL_NAME(new_decl));
  SET_DECL_RTL(new_decl, NULL);
  DECL_STATIC_CONSTRUCTOR(new_decl) = 0;
  DECL_STATIC_DESTRUCTOR(new_decl) = 0;
  new_node = old_node->create_version_clone(new_decl, vNULL, NULL);
  if (old_node->in_other_partition)
    new_node->in_other_partition = 1;

  if (new_node == NULL)
    return new_node;

  set_decl_built_in_function(new_node->decl, NOT_BUILT_IN, 0);
  TREE_PUBLIC(new_node->decl) = TREE_PUBLIC(old_node->decl);
  DECL_COMDAT(new_node->decl) = DECL_COMDAT(old_node->decl);
  DECL_WEAK(new_node->decl) = DECL_WEAK(old_node->decl);
  DECL_EXTERNAL(new_node->decl) = DECL_EXTERNAL(old_node->decl);
  DECL_VISIBILITY_SPECIFIED(new_node->decl) = DECL_VISIBILITY_SPECIFIED(old_node->decl);
  DECL_VISIBILITY(new_node->decl) = DECL_VISIBILITY(old_node->decl);
  DECL_DLLIMPORT_P(new_node->decl) = DECL_DLLIMPORT_P(old_node->decl);
  if (DECL_ONE_ONLY(old_node->decl))
    make_decl_one_only(new_node->decl, DECL_ASSEMBLER_NAME(new_node->decl));

  /* The method cgraph_version_clone_with_body () will force the new
     symbol local.  Undo this, and inherit external visibility from
     the old node.  */
  new_node->local = old_node->local;
  new_node->externally_visible = old_node->externally_visible;
#ifdef ZHAOCW_20250330_Fix
  new_node->calls_declare_variant_alt = old_node->calls_declare_variant_alt;
#else
  new_node->has_omp_variant_constructs = old_node->has_omp_variant_constructs;
#endif
  return new_node;
}

static void
find_call(function *fun)
{
  basic_block bb;

  FOR_EACH_BB_FN(bb, fun)
  {
    gimple_stmt_iterator i;

    for (i = gsi_start_bb(bb); !gsi_end_p(i);)
    {
      poly_uint64 vf = 1;
      enum internal_fn ifn;
      gimple *stmt = gsi_stmt(i);
      tree t;
      // if (!is_gimple_call(stmt) || !gimple_call_internal_p(stmt))
      if (is_gimple_call(stmt))
      {
        // gsi_next(&i);
        // continue;
        // 获取函数声明
        tree funcdecl = gimple_call_fndecl(stmt);
        if (funcdecl && TREE_CODE(funcdecl) == FUNCTION_DECL)
        {
          // 获取函数名
          const char *func_name = IDENTIFIER_POINTER(DECL_NAME(funcdecl));
          // 比较函数名
          if (strcmp(func_name, "foofoo") == 0)
          {

            // // 创建新的函数名
            // tree new_func_name = get_identifier("VFOOFOO");
            // DECL_NAME(funcdecl) = new_func_name;

            // modify call site
            //  tree new_fn = build_function_type_list(void_type_node, NULL_TREE);
            //   tree new_decl = build_function_decl(get_identifier("VFOOFOO"), new_fn);
            //   // Create the new function declaration.
            //   tree new_fndecl = build_function_decl(new_symbol, true);
            // Replace the old function with the new one
            // gimple_call_set_fndecl(stmt, new_fndecl);

            // 可选：根据需求进行更多的更新操作
            printf("Replaced function call to 'FOOFOO' with 'VFOOFOO'\n");

            tree attr = lookup_attribute("omp declare simd",
                                         DECL_ATTRIBUTES(funcdecl));
            struct cgraph_node *node = cgraph_node::get(funcdecl);

            do
            {
              /* Start with parsing the "omp declare simd" attribute(s).  */
              bool inbranch_clause_specified;
              struct cgraph_simd_clone *clone_info = simd_clone_clauses_extract(node, TREE_VALUE(attr), &inbranch_clause_specified);
              if (clone_info == NULL)
                continue;
              poly_uint64 orig_simdlen = clone_info->simdlen;
              tree base_type = simd_clone_compute_base_data_type(node, clone_info);
#ifdef ZHAOCW_20250330_Fix
              int count = targetm.simd_clone.compute_vecsize_and_simdlen(node, clone_info,
                                                                         base_type, 0);
#else
              int count = targetm.simd_clone.compute_vecsize_and_simdlen(node, clone_info,
                                                                         base_type, 0, true);
#endif                                                                                                                                                  
              if (count == 0)
                continue;

              /* Loop over all COUNT ISA variants, and if !INBRANCH_CLAUSE_SPECIFIED,
             also create one inbranch and one !inbranch clone of it.  */
              // for (int i = 0; i < count * 2; i++)
              // {
              int i = 0;
              struct cgraph_simd_clone *clone = clone_info;
              if (inbranch_clause_specified && (i & 1) != 0)
                continue;

              if (i != 0)
              {
                clone = simd_clone_struct_alloc(clone_info->nargs + ((i & 1) != 0));
                simd_clone_struct_copy(clone, clone_info);
                /* Undo changes targetm.simd_clone.compute_vecsize_and_simdlen
               and simd_clone_adjust_argument_types did to the first
               clone's info.  */
                clone->nargs -= clone_info->inbranch;
                clone->simdlen = orig_simdlen;
                /* And call the target hook again to get the right ISA.  */
#ifdef ZHAOCW_20250330_Fix
                targetm.simd_clone.compute_vecsize_and_simdlen(node, clone,
                                                               base_type,
                                                               i / 2);
#else
                targetm.simd_clone.compute_vecsize_and_simdlen(node, clone,
                                                               base_type,
                                                               i / 2, true);
#endif                                                                                                                              
                if ((i & 1) != 0)
                  clone->inbranch = 1;
              }

              /* simd_clone_mangle might fail if such a clone has been created
                 already.  */
              tree id = simd_clone_manual_mangle(node, clone);
              if (id == NULL_TREE)
              {
                if (i == 0)
                  clone->nargs += clone->inbranch;
                continue;
              }

              /* Only when we are sure we want to create the clone actually
                 clone the function (or definitions) or create another
                 extern FUNCTION_DECL (for prototypes without definitions).  */
              struct cgraph_node *n = manual_declare_create(node);
              if (n == NULL)
              {
                if (i == 0)
                  clone->nargs += clone->inbranch;
                continue;
              }

              n->simdclone = clone;
              clone->origin = node;
              clone->next_clone = NULL;
              if (node->simd_clones == NULL)
              {
                clone->prev_clone = n;
                node->simd_clones = n;
              }
              else
              {
                clone->prev_clone = node->simd_clones->simdclone->prev_clone;
                clone->prev_clone->simdclone->next_clone = n;
                node->simd_clones->simdclone->prev_clone = n;
              }
              symtab->change_decl_assembler_name(n->decl, id);
              gimple_call_set_fndecl(stmt, n->decl);
              // 创建新的函数名
              tree new_func_name = get_identifier("VFOOFOO");
              DECL_NAME(n->decl) = new_func_name;
              // }for
            } while ((attr = lookup_attribute("omp declare simd", TREE_CHAIN(attr))));
          }
        }
      }
      gsi_next(&i);
    }
  }
}

namespace
{

  const pass_data pass_data_manual_vect =
      {
          GIMPLE_PASS,    /* type */
          "manual_vect",  /* name */
          OPTGROUP_NONE,  /* optinfo_flags */
          TV_MANUAL_VECT, /* tv_id */
          0,              /* properties_required */
          0,              /* properties_provided */
          0,              /* properties_destroyed */
          0,              /* todo_flags_start */
          0,              /* todo_flags_finish */
  };

  class pass_manual_vect : public gimple_opt_pass
  {
    // private:
    //     bool has_traversed_functions; // 用于标记是否已经遍历过函数
  public:
    pass_manual_vect(gcc::context *ctxt)
        : gimple_opt_pass(pass_data_manual_vect, ctxt)
    {
    }

    virtual bool gate(function *fun)
    {
      return true;
    }

    virtual unsigned int execute(function *);

  }; // class pass_cleanup_eh

  unsigned int
  pass_manual_vect::execute(function *fun)
  {
    // find_call(fun);

    printf("manual_vect HELLO !!\n");
    return 1;
  }

} // anon namespace

gimple_opt_pass *
make_pass_manual_vect(gcc::context *ctxt)
{
  return new pass_manual_vect(ctxt);
}