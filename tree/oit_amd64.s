// func oneIfTrue(x, y float32) float32
TEXT ·oneIfTrue(SB),$0
    MOVQ 0x00000000, ret+8(FP) // default 0 to the return value
    MOVQ x+0(FP), BX // put first arg in the BX register
    SUBQ BX, y+8(FP) // subtract the second arg from the first
    CMPQ BX, 0x00000000 // see if the register is now 0
    JEQ ifeq // when equal, change the return value
    RET
ifeq:
    MOVQ 0x3f800000, ret+16(FP) // move 1 to the return value
