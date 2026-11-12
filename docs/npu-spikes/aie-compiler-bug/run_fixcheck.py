import sys, os
sys.path.insert(0, os.path.expanduser('~/develop/npu-studio-effects'))
S = os.path.dirname(os.path.abspath(__file__))
import npu.fused as F, npu.rvm.tail as T
F.KERNEL = os.path.join(S, 'fused_composite.cc'); T.KERNEL = os.path.join(S, 'rvm_tail.cc')
which, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ok = (F.selftest(w, h, 4, 10) if which == 'fused' else T.selftest(w, h, 4, 10))
print("PASS" if ok else "FAIL", which, w, h)
