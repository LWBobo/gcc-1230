/* Natural loop functions
   Copyright (C) 1987-2025 Free Software Foundation, Inc.

This file is part of GCC.

GCC is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3, or (at your option) any later
version.

GCC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with GCC; see the file COPYING3.  If not see
<http://www.gnu.org/licenses/>.  */

#ifndef GCC_CFGLOOP_H
#define GCC_CFGLOOP_H

#include "cfgloopmanip.h"

/* Structure to hold decision about unrolling/peeling.  */
enum lpt_dec
{
  LPT_NONE,
  LPT_UNROLL_CONSTANT,
  LPT_UNROLL_RUNTIME,
  LPT_UNROLL_STUPID
};

struct GTY (()) lpt_decision {
  enum lpt_dec decision;
  unsigned times;
};

/* The type of extend applied to an IV.  */
enum iv_extend_code
{
  IV_SIGN_EXTEND,
  IV_ZERO_EXTEND,
  IV_UNKNOWN_EXTEND
};

typedef generic_wide_int <fixed_wide_int_storage <WIDE_INT_MAX_INL_PRECISION> >
  bound_wide_int;

/* The structure describing a bound on number of iterations of a loop.  */

class GTY ((chain_next ("%h.next"))) nb_iter_bound {
public:
  /* The statement STMT is executed at most ...  */
  gimple *stmt;

  /* ... BOUND + 1 times (BOUND must be an unsigned constant).
     The + 1 is added for the following reasons:

     a) 0 would otherwise be unused, while we would need to care more about
        overflows (as MAX + 1 is sometimes produced as the estimate on number
	of executions of STMT).
     b) it is consistent with the result of number_of_iterations_exit.  */
  bound_wide_int bound;

  /* True if, after executing the statement BOUND + 1 times, we will
     leave the loop; that is, all the statements after it are executed at most
     BOUND times.  */
  bool is_exit;

  /* The next bound in the list.  */
  class nb_iter_bound *next;
};

/* Description of the loop exit.  */

struct GTY ((for_user)) loop_exit {
  /* The exit edge.  */
  edge e;

  /* Previous and next exit in the list of the exits of the loop.  */
  struct loop_exit *prev;
  struct loop_exit *next;

  /* Next element in the list of loops from that E exits.  */
  struct loop_exit *next_e;
};

struct loop_exit_hasher : ggc_ptr_hash<loop_exit>
{
  typedef edge compare_type;

  static hashval_t hash (loop_exit *);
  static bool equal (loop_exit *, edge);
  static void remove (loop_exit *);
};

typedef class loop *loop_p;

/* An integer estimation of the number of iterations.  Estimate_state
   describes what is the state of the estimation.  */
enum loop_estimation
{
  /* Estimate was not computed yet.  */
  EST_NOT_COMPUTED,
  /* Estimate is ready.  */
  EST_AVAILABLE,
  EST_LAST
};

/* The structure describing non-overflow control induction variable for
   loop's exit edge.  */
struct GTY ((chain_next ("%h.next"))) control_iv {
  tree base;
  tree step;
  struct control_iv *next;
};

/* Structure to hold information for each natural loop.  */
class GTY ((chain_next ("%h.next"))) loop {
public:
  /* Index into loops array.  Note indices will never be reused after loop
     is destroyed.  */
  int num;

  /* Number of loop insns.  */
  unsigned ninsns;

  /* Basic block of loop header.  */
  basic_block header;

  /* Basic block of loop latch.  */
  basic_block latch;

  /* For loop unrolling/peeling decision.  */
  struct lpt_decision lpt_decision;

  /* Average number of executed insns per iteration.  */
  unsigned av_ninsns;

  /* Number of blocks contained within the loop.  */
  unsigned num_nodes;

  /* Superloops of the loop, starting with the outermost loop.  */
  vec<loop_p, va_gc> *superloops;

  /* The first inner (child) loop or NULL if innermost loop.  */
  class loop *inner;

  /* Link to the next (sibling) loop.  */
  class loop *next;

  /* Auxiliary info specific to a pass.  */
  void *GTY ((skip (""))) aux;

  /* The number of times the latch of the loop is executed.  This can be an
     INTEGER_CST, or a symbolic expression representing the number of
     iterations like "N - 1", or a COND_EXPR containing the runtime
     conditions under which the number of iterations is non zero.

     Don't access this field directly: number_of_latch_executions
     computes and caches the computed information in this field.  */
  tree nb_iterations;

  /* An integer guaranteed to be greater or equal to nb_iterations.  Only
     valid if any_upper_bound is true.  */
  bound_wide_int nb_iterations_upper_bound;

  bound_wide_int nb_iterations_likely_upper_bound;

  /* An integer giving an estimate on nb_iterations.  Unlike
     nb_iterations_upper_bound, there is no guarantee that it is at least
     nb_iterations.  */
  bound_wide_int nb_iterations_estimate;

  /* If > 0, an integer, where the user asserted that for any
     I in [ 0, nb_iterations ) and for any J in
     [ I, min ( I + safelen, nb_iterations ) ), the Ith and Jth iterations
     of the loop can be safely evaluated concurrently.  */
  int safelen;

  /* Preferred vectorization factor for the loop if non-zero.  */
  int simdlen;
  /* ZHAOCW_20250329_TASK-SIMD */ 
  int tasksimd;
  /* ZHAOCW_20250329_TASK-SIMD END */

  /* Constraints are generally set by consumers and affect certain
     semantics of niter analyzer APIs.  Currently the APIs affected are
     number_of_iterations_exit* functions and their callers.  One typical
     use case of constraints is to vectorize possibly infinite loop:

       1) Compute niter->assumptions by calling niter analyzer API and
	  record it as possible condition for loop versioning.
       2) Clear buffered result of niter/scev analyzer.
       3) Set constraint LOOP_C_FINITE assuming the loop is finite.
       4) Analyze data references.  Since data reference analysis depends
	  on niter/scev analyzer, the point is that niter/scev analysis
	  is done under circumstance of LOOP_C_FINITE constraint.
       5) Version the loop with niter->assumptions computed in step 1).
       6) Vectorize the versioned loop in which niter->assumptions is
	  checked to be true.
       7) Update constraints in versioned loops so that niter analyzer
	  in following passes can use it.

     Note consumers are usually the loop optimizers and it is consumers'
     responsibility to set/clear constraints correctly.  Failing to do
     that might result in hard to track down bugs in niter/scev consumers.  */
  unsigned constraints;

  /* An integer estimation of the number of iterations.  Estimate_state
     describes what is the state of the estimation.  */
  ENUM_BITFIELD(loop_estimation) estimate_state : 8;

  unsigned any_upper_bound : 1;
  unsigned any_estimate : 1;
  unsigned any_likely_upper_bound : 1;

  /* True if the loop can be parallel.  */
  unsigned can_be_parallel : 1;

  /* True if -Waggressive-loop-optimizations warned about this loop
     already.  */
  unsigned warned_aggressive_loop_optimizations : 1;

  /* True if this loop should never be vectorized.  */
  unsigned dont_vectorize : 1;

  /* True if we should try harder to vectorize this loop.  */
  unsigned force_vectorize : 1;

  /* True if the loop is part of an oacc kernels region.  */
  unsigned in_oacc_kernels_region : 1;

  /* True if the loop is known to be finite.  This is a localized
     flag_finite_loops or similar pragmas state.  */
  unsigned finite_p : 1;

  /* The number of times to unroll the loop.  0 means no information given,
     just do what we always do.  A value of 1 means do not unroll the loop.
     A value of USHRT_MAX means unroll with no specific unrolling factor.
     Other values means unroll with the given unrolling factor.  */
  unsigned short unroll;

  /* If this loop was inlined the main clique of the callee which does
     not need remapping when copying the loop body.  */
  unsigned short owned_clique;

  /* For SIMD loops, this is a unique identifier of the loop, referenced
     by IFN_GOMP_SIMD_VF, IFN_GOMP_SIMD_LANE and IFN_GOMP_SIMD_LAST_LANE
     builtins.  */
  tree simduid;

  /* In loop optimization, it's common to generate loops from the original
     loop.  This field records the index of the original loop which can be
     used to track the original loop from newly generated loops.  This can
     be done by calling function get_loop (cfun, orig_loop_num).  Note the
     original loop could be destroyed for various reasons thus no longer
     exists, as a result, function call to get_loop returns NULL pointer.
     In this case, this field should not be used and needs to be cleared
     whenever possible.  */
  int orig_loop_num;

  /* Upper bound on number of iterations of a loop.  */
  class nb_iter_bound *bounds;

  /* Non-overflow control ivs of a loop.  */
  struct control_iv *control_ivs;

  /* Head of the cyclic list of the exits of the loop.  */
  struct loop_exit *exits;

  /* Number of iteration analysis data for RTL.  */
  class niter_desc *simple_loop_desc;

  /* For sanity checking during loop fixup we record here the former
     loop header for loops marked for removal.  Note that this prevents
     the basic-block from being collected but its index can still be
     reused.  */
  basic_block former_header;
};

/* Set if the loop is known to be infinite.  */
#define LOOP_C_INFINITE		(1 << 0)
/* Set if the loop is known to be finite without any assumptions.  */
#define LOOP_C_FINITE		(1 << 1)

/* Set C to the LOOP constraint.  */
inline void
loop_constraint_set (class loop *loop, unsigned c)
{
  loop->constraints |= c;
}

/* Clear C from the LOOP constraint.  */
inline void
loop_constraint_clear (class loop *loop, unsigned c)
{
  loop->constraints &= ~c;
}

/* Check if C is set in the LOOP constraint.  */
inline bool
loop_constraint_set_p (class loop *loop, unsigned c)
{
  return (loop->constraints & c) == c;
}

/* Flags for state of loop structure.  */
enum
{
  LOOPS_HAVE_PREHEADERS = 1,
  LOOPS_HAVE_SIMPLE_LATCHES = 2,
  LOOPS_HAVE_MARKED_IRREDUCIBLE_REGIONS = 4,
  LOOPS_HAVE_RECORDED_EXITS = 8,
  LOOPS_MAY_HAVE_MULTIPLE_LATCHES = 16,
  LOOP_CLOSED_SSA = 32,
  LOOPS_NEED_FIXUP = 64,
  LOOPS_HAVE_FALLTHRU_PREHEADERS = 128
};

#define LOOPS_NORMAL (LOOPS_HAVE_PREHEADERS | LOOPS_HAVE_SIMPLE_LATCHES \
		      | LOOPS_HAVE_MARKED_IRREDUCIBLE_REGIONS)
#define AVOID_CFG_MODIFICATIONS (LOOPS_MAY_HAVE_MULTIPLE_LATCHES)

/* Structure to hold CFG information about natural loops within a function.  */
struct GTY (()) loops {
  /* State of loops.  */
  int state;

  /* Array of the loops.  */
  vec<loop_p, va_gc> *larray;

  /* Maps edges to the list of their descriptions as loop exits.  Edges
     whose sources or destinations have loop_father == NULL (which may
     happen during the cfg manipulations) should not appear in EXITS.  */
  hash_table<loop_exit_hasher> *GTY(()) exits;

  /* Pointer to root of loop hierarchy tree.  */
  class loop *tree_root;
};

/* Loop recognition.  */
bool bb_loop_header_p (basic_block);
void init_loops_structure (struct function *, struct loops *, unsigned);
extern struct loops *flow_loops_find (struct loops *);
extern void disambiguate_loops_with_multiple_latches (void);
extern void flow_loops_free (struct loops *);
extern void flow_loops_dump (FILE *,
			     void (*)(const class loop *, FILE *, int), int);
extern void flow_loop_dump (const class loop *, FILE *,
			    void (*)(const class loop *, FILE *, int), int);
class loop *alloc_loop (void);
extern void flow_loop_free (class loop *);
int flow_loop_nodes_find (basic_block, class loop *);
unsigned fix_loop_structure (bitmap changed_bbs);
bool mark_irreducible_loops (void);
void release_recorded_exits (function *);
void record_loop_exits (void);
void rescan_loop_exit (edge, bool, bool);
void sort_sibling_loops (function *);

/* Loop data structure manipulation/querying.  */
extern void flow_loop_tree_node_add (class loop *, class loop *,
				     class loop * = NULL);
extern void flow_loop_tree_node_remove (class loop *);
extern bool flow_loop_nested_p	(const class loop *, const class loop *);
extern bool flow_bb_inside_loop_p (const class loop *, const_basic_block);
extern class loop * find_common_loop (class loop *, class loop *);
class loop *superloop_at_depth (class loop *, unsigned);
struct eni_weights;
extern int num_loop_insns (const class loop *);
extern int average_num_loop_insns (const class loop *);
extern unsigned get_loop_level (const class loop *);
extern bool loop_exit_edge_p (const class loop *, const_edge);
extern bool loop_exits_to_bb_p (class loop *, basic_block);
extern bool loop_exits_from_bb_p (class loop *, basic_block);
extern void mark_loop_exit_edges (void);
extern dump_user_location_t get_loop_location (class loop *loop);

/* Loops & cfg manipulation.  */
extern basic_block *get_loop_body (const class loop *);
extern unsigned get_loop_body_with_size (const class loop *, basic_block *,
					 unsigned);
extern basic_block *get_loop_body_in_dom_order (const class loop *);
extern basic_block *get_loop_body_in_bfs_order (const class loop *);
extern basic_block *get_loop_body_in_custom_order (const class loop *,
			       int (*) (const void *, const void *));
extern basic_block *get_loop_body_in_custom_order (const class loop *, void *,
			       int (*) (const void *, const void *, void *));

extern auto_vec<edge> get_loop_exit_edges (const class loop *, basic_block * = NULL);
extern edge single_exit (const class loop *);
extern edge single_likely_exit (class loop *loop, const vec<edge> &);
extern unsigned num_loop_branches (const class loop *);

extern edge loop_preheader_edge (const class loop *);
extern edge loop_latch_edge (const class loop *);

extern void add_bb_to_loop (basic_block, class loop *);
extern void remove_bb_from_loops (basic_block);

extern void cancel_loop_tree (class loop *);
extern void delete_loop (class loop *);


extern void verify_loop_structure (void);

/* Loop analysis.  */
extern bool just_once_each_iteration_p (const class loop *, const_basic_block);
gcov_type expected_loop_iterations_unbounded (const class loop *,
					      bool *read_profile_p = NULL);
extern bool expected_loop_iterations_by_profile (const class loop *loop,
						 sreal *ret,
						 bool *reliable = NULL);
extern bool maybe_flat_loop_profile (const class loop *);
extern unsigned expected_loop_iterations (class loop *);
extern rtx doloop_condition_get (rtx_insn *);

void mark_loop_for_removal (loop_p);
void print_loop_info (FILE *file, const class loop *loop, const char *);

/* Induction variable analysis.  */

/* The description of induction variable.  The things are a bit complicated
   due to need to handle subregs and extends.  The value of the object described
   by it can be obtained as follows (all computations are done in extend_mode):

   Value in i-th iteration is
     delta + mult * extend_{extend_mode} (subreg_{mode} (base + i * step)).

   If first_special is true, the value in the first iteration is
     delta + mult * base

   If extend = UNKNOWN, first_special must be false, delta 0, mult 1 and value is
     subreg_{mode} (base + i * step)

   The get_iv_value function can be used to obtain these expressions.

   ??? Add a third mode field that would specify the mode in that inner
   computation is done, which would enable it to be different from the
   outer one?  */

class rtx_iv
{
public:
  /* Its base and step (mode of base and step is supposed to be extend_mode,
     see the description above).  */
  rtx base, step;

  /* The type of extend applied to it (IV_SIGN_EXTEND, IV_ZERO_EXTEND,
     or IV_UNKNOWN_EXTEND).  */
  enum iv_extend_code extend;

  /* Operations applied in the extended mode.  */
  rtx delta, mult;

  /* The mode it is extended to.  */
  scalar_int_mode extend_mode;

  /* The mode the variable iterates in.  */
  scalar_int_mode mode;

  /* Whether the first iteration needs to be handled specially.  */
  unsigned first_special : 1;
};

/* The description of an exit from the loop and of the number of iterations
   till we take the exit.  */

class GTY(()) niter_desc
{
public:
  /* The edge out of the loop.  */
  edge out_edge;

  /* The other edge leading from the condition.  */
  edge in_edge;

  /* True if we are able to say anything about number of iterations of the
     loop.  */
  bool simple_p;

  /* True if the loop iterates the constant number of times.  */
  bool const_iter;

  /* Number of iterations if constant.  */
  uint64_t niter;

  /* Assumptions under that the rest of the information is valid.  */
  rtx assumptions;

  /* Assumptions under that the loop ends before reaching the latch,
     even if value of niter_expr says otherwise.  */
  rtx noloop_assumptions;

  /* Condition under that the loop is infinite.  */
  rtx infinite;

  /* Whether the comparison is signed.  */
  bool signed_p;

  /* The mode in that niter_expr should be computed.  */
  scalar_int_mode mode;

  /* The number of iterations of the loop.  */
  rtx niter_expr;
};

extern void iv_analysis_loop_init (class loop *);
extern bool iv_analyze (rtx_insn *, scalar_int_mode, rtx, class rtx_iv *);
extern bool iv_analyze_result (rtx_insn *, rtx, class rtx_iv *);
extern bool iv_analyze_expr (rtx_insn *, scalar_int_mode, rtx,
			     class rtx_iv *);
extern rtx get_iv_value (class rtx_iv *, rtx);
extern bool biv_p (rtx_insn *, scalar_int_mode, rtx);
extern void iv_analysis_done (void);

extern class niter_desc *get_simple_loop_desc (class loop *loop);
extern void free_simple_loop_desc (class loop *loop);

inline class niter_desc *
simple_loop_desc (class loop *loop)
{
  return loop->simple_loop_desc;
}

/* Accessors for the loop structures.  */

/* Returns the loop with index NUM from FNs loop tree.  */

inline class loop *
get_loop (struct function *fn, unsigned num)
{
  return (*loops_for_fn (fn)->larray)[num];
}

/* Returns the number of superloops of LOOP.  */

inline unsigned
loop_depth (const class loop *loop)
{
  return vec_safe_length (loop->superloops);
}

/* Returns the immediate superloop of LOOP, or NULL if LOOP is the outermost
   loop.  */

inline class loop *
loop_outer (const class loop *loop)
{
  unsigned n = vec_safe_length (loop->superloops);

  if (n == 0)
    return NULL;

  return (*loop->superloops)[n - 1];
}

/* Returns true if LOOP has at least one exit edge.  */

inline bool
loop_has_exit_edges (const class loop *loop)
{
  return loop->exits->next->e != NULL;
}

/* Returns the list of loops in FN.  */

inline vec<loop_p, va_gc> *
get_loops (struct function *fn)
{
  struct loops *loops = loops_for_fn (fn);
  if (!loops)
    return NULL;

  return loops->larray;
}

/* Returns the number of loops in FN (including the removed
   ones and the fake loop that forms the root of the loop tree).  */

inline unsigned
number_of_loops (struct function *fn)
{
  struct loops *loops = loops_for_fn (fn);
  if (!loops)
    return 0;

  return vec_safe_length (loops->larray);
}

/* Returns true if state of the loops satisfies all properties
   described by FLAGS.  */

inline bool
loops_state_satisfies_p (function *fn, unsigned flags)
{
  return (loops_for_fn (fn)->state & flags) == flags;
}

inline bool
loops_state_satisfies_p (unsigned flags)
{
  return loops_state_satisfies_p (cfun, flags);
}

/* Sets FLAGS to the loops state.  */

inline void
loops_state_set (function *fn, unsigned flags)
{
  loops_for_fn (fn)->state |= flags;
}

inline void
loops_state_set (unsigned flags)
{
  loops_state_set (cfun, flags);
}

/* Clears FLAGS from the loops state.  */

inline void
loops_state_clear (function *fn, unsigned flags)
{
  loops_for_fn (fn)->state &= ~flags;
}

inline void
loops_state_clear (unsigned flags)
{
  if (!current_loops)
    return;
  loops_state_clear (cfun, flags);
}

/* Check loop structure invariants, if internal consistency checks are
   enabled.  */

inline void
checking_verify_loop_structure (void)
{
  /* VERIFY_LOOP_STRUCTURE essentially asserts that no loops need fixups.

     The loop optimizers should never make changes to the CFG which
     require loop fixups.  But the low level CFG manipulation code may
     set the flag conservatively.

     Go ahead and clear the flag here.  That avoids the assert inside
     VERIFY_LOOP_STRUCTURE, and if there is an inconsistency in the loop
     structures VERIFY_LOOP_STRUCTURE will detect it.

     This also avoid the compile time cost of excessive fixups.  */
  loops_state_clear (LOOPS_NEED_FIXUP);
  if (flag_checking)
    verify_loop_structure ();
}

/* Loop iterators.  */

/* Flags for loop iteration.  */

enum li_flags
{
  LI_INCLUDE_ROOT = 1,		/* Include the fake root of the loop tree.  */
  LI_FROM_INNERMOST = 2,	/* Iterate over the loops in the reverse order,
				   starting from innermost ones.  */
  LI_ONLY_INNERMOST = 4		/* Iterate only over innermost loops.  */
};

/* Provide the functionality of std::as_const to support range-based for
   to use const iterator.  (We can't use std::as_const itself because it's
   a C++17 feature.)  */
template <typename T>
constexpr const T &
as_const (T &t)
{
  return t;
}

/* A list for visiting loops, which contains the loop numbers instead of
   the loop pointers.  If the loop ROOT is offered (non-null), the visiting
   will start from it, otherwise it would start from the tree_root of
   loops_for_fn (FN) instead.  The scope is restricted in function FN and
   the visiting order is specified by FLAGS.  */

class loops_list
{
public:
  loops_list (function *fn, unsigned flags, class loop *root = nullptr);

  template <typename T> class Iter
  {
  public:
    Iter (const loops_list &l, unsigned idx) : list (l), curr_idx (idx)
    {
      fill_curr_loop ();
    }

    T operator* () const { return curr_loop; }

    Iter &
    operator++ ()
    {
      if (curr_idx < list.to_visit.length ())
	{
	  /* Bump the index and fill a new one.  */
	  curr_idx++;
	  fill_curr_loop ();
	}
      else
	gcc_assert (!curr_loop);

      return *this;
    }

    bool
    operator!= (const Iter &rhs) const
    {
      return this->curr_idx != rhs.curr_idx;
    }

  private:
    /* Fill the current loop starting from the current index.  */
    void fill_curr_loop ();

    /* Reference to the loop list to visit.  */
    const loops_list &list;

    /* The current index in the list to visit.  */
    unsigned curr_idx;

    /* The loop implied by the current index.  */
    class loop *curr_loop;
  };

  using iterator = Iter<class loop *>;
  using const_iterator = Iter<const class loop *>;

  iterator
  begin ()
  {
    return iterator (*this, 0);
  }

  iterator
  end ()
  {
    return iterator (*this, to_visit.length ());
  }

  const_iterator
  begin () const
  {
    return const_iterator (*this, 0);
  }

  const_iterator
  end () const
  {
    return const_iterator (*this, to_visit.length ());
  }

private:
  /* Walk loop tree starting from ROOT as the visiting order specified
     by FLAGS.  */
  void walk_loop_tree (class loop *root, unsigned flags);

  /* The function we are visiting.  */
  function *fn;

  /* The list of loops to visit.  */
  auto_vec<int, 16> to_visit;
};

/* Starting from current index CURR_IDX (inclusive), find one index
   which stands for one valid loop and fill the found loop as CURR_LOOP,
   if we can't find one, set CURR_LOOP as null.  */

template <typename T>
inline void
loops_list::Iter<T>::fill_curr_loop ()
{
  int anum;

  while (this->list.to_visit.iterate (this->curr_idx, &anum))
    {
      class loop *loop = get_loop (this->list.fn, anum);
      if (loop)
	{
	  curr_loop = loop;
	  return;
	}
      this->curr_idx++;
    }

  curr_loop = nullptr;
}

/* Set up the loops list to visit according to the specified
   function scope FN and iterating order FLAGS.  If ROOT is
   not null, the visiting would start from it, otherwise it
   will start from tree_root of loops_for_fn (FN).  */

inline loops_list::loops_list (function *fn, unsigned flags, class loop *root)
{
  struct loops *loops = loops_for_fn (fn);
  gcc_assert (!root || loops);

  /* Check mutually exclusive flags should not co-exist.  */
  unsigned checked_flags = LI_ONLY_INNERMOST | LI_FROM_INNERMOST;
  gcc_assert ((flags & checked_flags) != checked_flags);

  this->fn = fn;
  if (!loops)
    return;

  class loop *tree_root = root ? root : loops->tree_root;

  this->to_visit.reserve_exact (number_of_loops (fn));

  /* When root is tree_root of loops_for_fn (fn) and the visiting
     order is LI_ONLY_INNERMOST, we would like to use linear
     search here since it has a more stable bound than the
     walk_loop_tree.  */
  if (flags & LI_ONLY_INNERMOST && tree_root == loops->tree_root)
    {
      gcc_assert (tree_root->num == 0);
      if (tree_root->inner == NULL)
	{
	  if (flags & LI_INCLUDE_ROOT)
	    this->to_visit.quick_push (0);

	  return;
	}

      class loop *aloop;
      unsigned int i;
      for (i = 1; vec_safe_iterate (loops->larray, i, &aloop); i++)
	if (aloop != NULL && aloop->inner == NULL)
	  this->to_visit.quick_push (aloop->num);
    }
  else
    walk_loop_tree (tree_root, flags);
}

/* The properties of the target.  */
struct target_cfgloop {
  /* Number of available registers.  */
  unsigned x_target_avail_regs;

  /* Number of available registers that are call-clobbered.  */
  unsigned x_target_clobbered_regs;

  /* Number of registers reserved for temporary expressions.  */
  unsigned x_target_res_regs;

  /* The cost for register when there still is some reserve, but we are
     approaching the number of available registers.  */
  unsigned x_target_reg_cost[2];

  /* The cost for register when we need to spill.  */
  unsigned x_target_spill_cost[2];
};

extern struct target_cfgloop default_target_cfgloop;
#if SWITCHABLE_TARGET
extern struct target_cfgloop *this_target_cfgloop;
#else
#define this_target_cfgloop (&default_target_cfgloop)
#endif

#define target_avail_regs \
  (this_target_cfgloop->x_target_avail_regs)
#define target_clobbered_regs \
  (this_target_cfgloop->x_target_clobbered_regs)
#define target_res_regs \
  (this_target_cfgloop->x_target_res_regs)
#define target_reg_cost \
  (this_target_cfgloop->x_target_reg_cost)
#define target_spill_cost \
  (this_target_cfgloop->x_target_spill_cost)

/* Register pressure estimation for induction variable optimizations & loop
   invariant motion.  */
extern unsigned estimate_reg_pressure_cost (unsigned, unsigned, bool, bool);
extern void init_set_costs (void);

/* Loop optimizer initialization.  */
extern void loop_optimizer_init (unsigned);
extern void loop_optimizer_finalize (function *, bool = false);
inline void
loop_optimizer_finalize ()
{
  loop_optimizer_finalize (cfun);
}

/* Optimization passes.  */
enum
{
  UAP_UNROLL = 1,	/* Enables unrolling of loops if it seems profitable.  */
  UAP_UNROLL_ALL = 2	/* Enables unrolling of all loops.  */
};

extern void doloop_optimize_loops (void);
extern void move_loop_invariants (void);
extern auto_vec<basic_block> get_loop_hot_path (const class loop *loop);

/* Returns the outermost loop of the loop nest that contains LOOP.*/
inline class loop *
loop_outermost (class loop *loop)
{
  unsigned n = vec_safe_length (loop->superloops);

  if (n <= 1)
    return loop;

  return (*loop->superloops)[1];
}

extern void record_niter_bound (class loop *, const widest_int &, bool, bool);
extern HOST_WIDE_INT get_estimated_loop_iterations_int (class loop *);
extern HOST_WIDE_INT get_max_loop_iterations_int (const class loop *);
extern HOST_WIDE_INT get_likely_max_loop_iterations_int (class loop *);
extern bool get_estimated_loop_iterations (class loop *loop, widest_int *nit);
extern bool get_max_loop_iterations (const class loop *loop, widest_int *nit);
extern bool get_likely_max_loop_iterations (class loop *loop, widest_int *nit);
extern int bb_loop_depth (const_basic_block);
extern edge single_dom_exit (class loop *);
extern profile_count loop_count_in (const class loop *loop);

/* Converts VAL to widest_int.  */

inline widest_int
gcov_type_to_wide_int (gcov_type val)
{
  HOST_WIDE_INT a[2];

  a[0] = (unsigned HOST_WIDE_INT) val;
  /* If HOST_BITS_PER_WIDE_INT == HOST_BITS_PER_WIDEST_INT, avoid shifting by
     the size of type.  */
  val >>= HOST_BITS_PER_WIDE_INT - 1;
  val >>= 1;
  a[1] = (unsigned HOST_WIDE_INT) val;

  return widest_int::from_array (a, 2);
}
#endif /* GCC_CFGLOOP_H */
