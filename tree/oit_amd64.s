// func oneIfTrue(x, y float32) float32
TEXT ·oneIfTrue(SB),$0
    MOVQ 0.0, ret+8(FP) // default 0 to the return value
    MOVQ val1+0(FP), BX // put first arg in the BX register
    SUBQ BX, val2+4(FP) // subtract the second arg from the first
    CMPQ BX, 0x0000 // see if the register is now 0
    JEQ ifeq // when equal, change the return value
    RET
ifeq:
    MOVQ 0x0001, ret+8(FP) // move 1 to the return value
