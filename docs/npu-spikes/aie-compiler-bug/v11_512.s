	.file	"probe.cc"
	.section	.text.probeLine,"ax",@progbits
	.globl	probeLine                       // -- Begin function probeLine
	.p2align	4
	.type	probeLine,@function
probeLine:                              // @probeLine
// %bb.0:                               // %entry
	nopb	;		nopa	;		nops	;		movxm	p3, #.L__const._ZN6rowops3Up44makeEv.p;		nopv	
	mova	r2, #32;		paddb	[sp], #32;		movxm	p5, #.L__const._ZN6rowops3Up44makeEv.n;		nops	
	ltu	r27, r2, r0;		mov	p2, sp
	mova	r1, #512;		paddb	[p2], #-32;		sel.nez	r2, r0, r2, r27;		mov	p4, p3
	mova	r3, #-5;		st	r1, [p2, #0];		add	r2, r2, #-1
	lda	r1, [p2, #0];		paddb	[p4], #32;		movxm	p2, #.L__const._ZN6rowops3Up44makeEv.c
	vldb	wh0, [p4, #0];		lshl	r3, r2, r3;		mov	p4, p2
	vldb	wl2, [p2, #0];		add	r3, r3, #1;		mov	p2, p5
	mova	r3, #1;		paddb	[p2], #32;		movx	r6, #0;		add.nc	lc, r3, #-1
	vldb	wh4, [p2, #0];		lshl	r5, r6, r3;		mov	p2, p0
	mov	m1, r5
	mova	m0, #-30;		paddb	[p2], m1
	vlda	wl6, [p2], #1056;		vldb	wl0, [p3, #0];		mov	p3, p2
	vldb	wh6, [p2], m0
	vldb	wl8, [p2], #32;		mov	r5, p2
	vldb	wh8, [p2], m0
	nop	
	padda	[p3], #1024
	mova	r4, #31;		vldb	wl10, [p2, #0];		mov	r7, p3
	vldb	wh10, [p2, #32];		and	r7, r7, r4;		mov	s0, #3
	mova	r1, #18;		vshift	x6, x6, x0, r7
	paddb	[p4], #32;		and	r5, r5, r4;		vshuffle	x6, x6, x6, r1
	vldb	wh2, [p4, #0];		vshift	x8, x8, x0, r5
	mova	r0, #16;		mov	r5, p2
	vldb	wl4, [p5, #0];		vshuffle	x6, x6, x6, r0
	and	r5, r5, r4;		vshuffle	x8, x8, x8, r1
	mova	r2, #824;		vshift	x10, x10, x0, r5
	vshuffle	x6, x8, x8, r0;		vmul	cm0, x6, x0, r2
	vshuffle	x8, x10, x10, r1
	vshuffle	x6, x8, x8, r0;		vmac	cm0, cm0, x6, x2, r2
	movxm	ls, #.LBB0_1
	movxm	le, #.L_LEnd0;		vmac	cm2, cm0, x6, x4, r2
	lshl	r7, r6, r3;		mov	crSat, #1
	mov	m1, r7
	add	r5, r6, #32;		mov	p2, p1
	paddb	[p2], m1;		add	r6, r6, #8;		mov	crRnd, #0
.LBB0_1:                                // %for.body
                                        // =>This Inner Loop Header: Depth=1
	nopa	;		vst.srs.s16.s32	bmh2, s0, [p2, #32];		lshl	r7, r6, r3;		mov	p3, p0
	vst.srs.s16.s32	bml2, s0, [p2, #0];		mov	m1, r7
	paddb	[p3], m1
	mov	p2, p3
	vlda	wl6, [p2], #1056
	vldb	wh6, [p2], m0
	vldb	wl8, [p2], #32;		mov	r7, p2
	vldb	wh8, [p2], m0
	nop	
	padda	[p3], #1024
	vldb	wh10, [p2, #32];		mov	r8, p3
	vldb	wl10, [p2, #0];		and	r8, r8, r4
	vshift	x1, x6, x0, r8
	and	r7, r7, r4;		vshuffle	x3, x1, x1, r1
	vshift	x5, x8, x0, r7
	mov	r7, p2
	vshuffle	x9, x3, x3, r0
	and	r7, r7, r4;		vshuffle	x7, x5, x5, r1
	vshift	x11, x10, x0, r7
	vshuffle	x7, x7, x7, r0;		vmul	cm0, x9, x0, r2
	vshuffle	x11, x11, x11, r1
	vshuffle	x11, x11, x11, r0;		vmac	cm1, cm0, x7, x2, r2
	nop	
	vmac	cm2, cm1, x11, x4, r2
	nop	
	lshl	r7, r5, r3;		mov	p2, p1
	add	r6, r6, #8;		mov	m1, r7
.L_LEnd0:
	paddb	[p2], m1;		nopa	;		nops	;		add	r5, r5, #32;		nopm	;		nopv	
// %bb.2:
	nopb	;		nopa	;		nops	;		ret lr	;		nopm	;		nopv	
	vst.srs.s16.s32	bmh2, s0, [p2, #32]     //  Delay Slot 5
	vst.srs.s16.s32	bml2, s0, [p2, #0]      //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	paddb	[sp], #-32                      //  Delay Slot 1
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
