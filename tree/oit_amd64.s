// +build amd64 !noasm !appengine

#include "textflag.h"

// func oneIfTrue(x, y float32) float32
TEXT ·oneIfTrue(SB),NOSPLIT,$0
    MOVSS x+0(FP), X0 // put first arg in a register
    MOVSS y+4(FP), X1  // put second arg in a register
    UCOMISS X0, X1
    JNE NOTEQ // when not equal return 0
    MOVQ $0x3f800000, ret+8(FP) // when equal return 0
    RET
NOTEQ:
    MOVQ $0x00000000, ret+8(FP) // move 0 to the return value
    RET
