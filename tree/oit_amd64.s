// +build amd64 !noasm !appengine

#include "textflag.h"

// func oneIfTrue(x, y float32) float32
TEXT ·oneIfTrue(SB),$8
    MOVQ $0, ret+8(FP) // default 0 to the return value
    MOVQ x+0(FP), BX // put first arg in the BX register
    SUBQ BX, y+4(FP) // subtract the second arg from the first
    CMPQ BX, $0 // see if the register is now 0
    JEQ ifeq // when equal, change the return value
    RET
ifeq:
    MOVQ $1, ret+8(FP) // move 1 to the return value
