	.file	"probe.cc"
	.section	.text.probeLine,"ax",@progbits
	.globl	probeLine                       // -- Begin function probeLine
	.p2align	4
	.type	probeLine,@function
probeLine:                              // @probeLine
// %bb.0:                               // %entry
	paddb	[sp], #32;		nopx	
	mov	p2, sp
	mova	r1, #496;		paddb	[p2], #-32;		movxm	p3, #.L__const._ZN6rowops3Up44makeEv.c
	st	r1, [p2, #0];		movxm	p5, #.L__const._ZN6rowops3Up44makeEv.n
	lda	r1, [p2, #0];		movx	r4, #32
	ltu	r27, r4, r0
	sel.nez	r4, r0, r4, r27
	movxm	p2, #.L__const._ZN6rowops3Up44makeEv.p
	mova	r5, #-5;		add	r4, r4, #-1;		mov	p4, p2
	vldb	wl0, [p2, #0];		lshl	r4, r4, r5;		mov	p2, p5
	paddb	[p2], #32;		add	r5, r4, #1
	mova	r1, #0;		vldb	wh4, [p2, #0];		movx	r4, #1;		add.nc	lc, r5, #-1
	mova	m0, #-30;		lshl	r7, r1, r4;		mov	p2, p0
	paddb	[p4], #32;		add	r1, r1, #8;		mov	m1, r7
	vlda	wl2, [p3, #0];		vldb	wh0, [p4, #0];		padds	[p2], m1;		mov	p4, p3
	vlda	wl6, [p2, #992];		padds	[p2], #992;		mov	p3, p2
	padda	[p3], #1024;		movx	r5, #31;		mov	r8, p2
	vldb	wh6, [p3], m0;		and	r8, r8, r5;		mov	p2, p1
	vldb	wl8, [p3], #32;		mov	r7, p3
	vldb	wh8, [p3], m0;		and	r7, r7, r5
	vlda	wl10, [p3, #0]
	vldb	wh10, [p3, #32]
	nop	
	nop	
	mova	r2, #18;		vshift	x1, x6, x0, r8
	paddb	[p4], #32;		vshuffle	x7, x1, x1, r2
	vldb	wh2, [p4, #0];		vshift	x3, x8, x0, r7
	mova	r0, #16;		mov	r7, p3
	mova	r6, #0;		vldb	wl4, [p5, #0];		and	r7, r7, r5;		vshuffle	x9, x7, x7, r0
	mova	r3, #824;		lshl	r7, r6, r4;		vshift	x5, x10, x0, r7
	add	r6, r6, #32;		vshuffle	x11, x3, x3, r2;		vmul	cm0, x9, x0, r3
	vshuffle	x11, x11, x11, r0
	vshuffle	x5, x5, x5, r2
	vshuffle	x5, x5, x5, r0;		vmac	cm1, cm0, x11, x2, r3
	mov	m1, r7
	padda	[p2], m1;		vmac	cm2, cm1, x5, x4, r3
	movxm	ls, #.LBB0_1
	mov	s0, #3
	mov	crSat, #1
	mov	crRnd, #0
	vst.srs.s16.s32	bmh2, s0, [p2, #32];		movxm	le, #.L_LEnd0
.LBB0_1:                                // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopb	;		nopa	;		vst.srs.s16.s32	bml2, s0, [p2, #0];		lshl	r7, r1, r4;		mov	p2, p0;		nopv	
	nopa	;		nopb	;		add	r1, r1, #8;		mov	m1, r7
	padds	[p2], m1
	vlda	wl6, [p2, #992];		padds	[p2], #992;		mov	p3, p2
	padda	[p3], #1024;		mov	r8, p2
	vldb	wh6, [p3], m0;		and	r8, r8, r5;		mov	p2, p1
	vldb	wl8, [p3], #32;		mov	r7, p3
	vldb	wh8, [p3], m0;		and	r7, r7, r5
	vlda	wl10, [p3, #0]
	vldb	wh10, [p3, #32]
	nop	
	nop	
	vshift	x1, x6, x0, r8
	vshuffle	x7, x1, x1, r2
	vshift	x3, x8, x0, r7
	mov	r7, p3
	and	r7, r7, r5;		vshuffle	x9, x7, x7, r0
	lshl	r7, r6, r4;		vshift	x5, x10, x0, r7
	add	r6, r6, #32;		vshuffle	x11, x3, x3, r2;		vmul	cm0, x9, x0, r3
	vshuffle	x11, x11, x11, r0
	vshuffle	x5, x5, x5, r2
	vshuffle	x5, x5, x5, r0;		vmac	cm1, cm0, x11, x2, r3
	mov	m1, r7
	padda	[p2], m1;		vmac	cm2, cm1, x5, x4, r3
	nop	
	nop	
	nop	
	nop	
.L_LEnd0:
	nopb	;		nopa	;		vst.srs.s16.s32	bmh2, s0, [p2, #32];		nopxm	;		nopv	
// %bb.2:                               // %for.cond.cleanup
	paddb	[sp], #-32;		nopa	;		vst.srs.s16.s32	bml2, s0, [p2, #0];		ret lr	;		nopm	;		nopv	
	nopa	;		nopb	;		nopx	 //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end0:
	.size	probeLine, .Lfunc_end0-probeLine
                                        // -- End function
	.type	.L__const._ZN6rowops3Up44makeEv.p,@object // @__const._ZN6rowops3Up44makeEv.p
	.section	.rodata..L__const._ZN6rowops3Up44makeEv.p,"a",@progbits
	.p2align	6, 0x0
.L__const._ZN6rowops3Up44makeEv.p:
	.short	3                               // 0x3
	.short	1                               // 0x1
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	3                               // 0x3
	.short	1                               // 0x1
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	3                               // 0x3
	.short	1                               // 0x1
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	3                               // 0x3
	.short	1                               // 0x1
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	3                               // 0x3
	.short	1                               // 0x1
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	3                               // 0x3
	.short	1                               // 0x1
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	3                               // 0x3
	.short	1                               // 0x1
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	3                               // 0x3
	.short	1                               // 0x1
	.short	0                               // 0x0
	.short	0                               // 0x0
	.size	.L__const._ZN6rowops3Up44makeEv.p, 64

	.type	.L__const._ZN6rowops3Up44makeEv.c,@object // @__const._ZN6rowops3Up44makeEv.c
	.section	.rodata..L__const._ZN6rowops3Up44makeEv.c,"a",@progbits
	.p2align	6, 0x0
.L__const._ZN6rowops3Up44makeEv.c:
	.short	5                               // 0x5
	.short	7                               // 0x7
	.short	7                               // 0x7
	.short	5                               // 0x5
	.short	5                               // 0x5
	.short	7                               // 0x7
	.short	7                               // 0x7
	.short	5                               // 0x5
	.short	5                               // 0x5
	.short	7                               // 0x7
	.short	7                               // 0x7
	.short	5                               // 0x5
	.short	5                               // 0x5
	.short	7                               // 0x7
	.short	7                               // 0x7
	.short	5                               // 0x5
	.short	5                               // 0x5
	.short	7                               // 0x7
	.short	7                               // 0x7
	.short	5                               // 0x5
	.short	5                               // 0x5
	.short	7                               // 0x7
	.short	7                               // 0x7
	.short	5                               // 0x5
	.short	5                               // 0x5
	.short	7                               // 0x7
	.short	7                               // 0x7
	.short	5                               // 0x5
	.short	5                               // 0x5
	.short	7                               // 0x7
	.short	7                               // 0x7
	.short	5                               // 0x5
	.size	.L__const._ZN6rowops3Up44makeEv.c, 64

	.type	.L__const._ZN6rowops3Up44makeEv.n,@object // @__const._ZN6rowops3Up44makeEv.n
	.section	.rodata..L__const._ZN6rowops3Up44makeEv.n,"a",@progbits
	.p2align	6, 0x0
.L__const._ZN6rowops3Up44makeEv.n:
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	1                               // 0x1
	.short	3                               // 0x3
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	1                               // 0x1
	.short	3                               // 0x3
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	1                               // 0x1
	.short	3                               // 0x3
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	1                               // 0x1
	.short	3                               // 0x3
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	1                               // 0x1
	.short	3                               // 0x3
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	1                               // 0x1
	.short	3                               // 0x3
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	1                               // 0x1
	.short	3                               // 0x3
	.short	0                               // 0x0
	.short	0                               // 0x0
	.short	1                               // 0x1
	.short	3                               // 0x3
	.size	.L__const._ZN6rowops3Up44makeEv.n, 64

	.section	".linker-options","e",@llvm_linker_options
	.ident	"clang version 21.0.0 (https://github.com/Xilinx/llvm-aie cb664e8cc3eb42a12e3ad3cee28729785ffa97a3)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
