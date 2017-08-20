// +build amd64 !noasm !appengine

#include "textflag.h"

// func oneIfTrue(x, y float32) float32
TEXT ·oneIfTrue(SB),NOSPLIT,$0
    MOVQ x+0(FP), CX // put first arg in the CX register
    MOVQ y+4(FP), SI  // put second arg in SI register
    CMPQ CX, SI // see if first arg equals second arg
    JE RONE // when equal, change the return value
    MOVQ $0, ret+8(FP) // default 0 as the return value
    RET
RONE:
    MOVQ $1, ret+8(FP) // move 1 to the return value
    RET
