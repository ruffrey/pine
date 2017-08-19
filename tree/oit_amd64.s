// func oneIfTrue(val1, val2 float32) (inc float32)
TEXT ·oneIfTrue(SB),$0
    MOVQ val1+0(FP), BX
    MOVQ val2+4(FP), CX
    SUBQ BX, CX
    CMPQ BX, 0
    JE ifeq
    RET
ifeq:
    MOVQ 0x01, ret+8(FP) // move 1 to the return value
