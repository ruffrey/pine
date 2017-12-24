// +build amd64 !noasm !appengine

#include "textflag.h"

// func oneIfTrue(x, y float32) float32
TEXT ·oneIfTrue(SB),NOSPLIT,$0
    MOVSS x+0(FP), X0 // put first arg in a register
    MOVSS y+4(FP), X1  // put second arg in a register
    UCOMISS X0, X1
    JE REONE // jump if equal
    MOVQ $0x00000000, ret+8(FP) // return 0 when not equal
    RET
REONE:
    MOVQ $0x3f800000, ret+8(FP) // move 1 to return
    RET
