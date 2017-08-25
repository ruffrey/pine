// +build amd64 !noasm !appengine

#include "textflag.h"

//
// memory layout of the stack relative to FP
//  +0 data slice ptr
//  +8 data slice len
// +16 data slice cap
// +24 splitGroup[0]  | splitGroup[1]
// +32 splitGroup[2]  | splitGroup[3]
// +40 splitGroup[4]  | splitGroup[5]

// func withValue(lastColIndex int, value float32, splitGroup []datarow) (count float32)
TEXT ·withValue(SB),NOSPLIT,$0
    MOVSS $0x00000000, X0 // this will be the return count
    MOVSS $0x3f800000, X1 // this is just the value of 1 and will be incremented into X0
    MOVQ lastColIndex+0(FP), X3 // save the first argument which is index of the last column
    MULQ $4, X3 // float32 is 4 bytes, so to point at the last column index, we multiply by 4 bytes
    MOVQ AX, X3
    MOVQ value+8(FP), X2 // the value we will compare against
    // data ptr
    MOVQ splitGroup+12(FP), CX
    // data len
    MOVQ splitGroup+20(FP), SI
    // index for the loop
    MOVQ $0, AX
    // return early if zero length
    CMPQ AX, SI
    JE END
LOOP:
    MOVSS (X3)(CX), X4 // put the last column of the row at the data pointer into register X4
    UCOMISS X4, X2
    JMP NEXT
IFEQ:
    ADDPS X0, X1 // they are the same - increment the return count by one
    JMP NEXT
NEXT:
    ADDQ $4, CX  // data pointer += 4 bytes
    INCQ AX // loop i++
    CMPQ AX, SI
    JLT LOOP
END:
    // put our final count into the return value, which is the very end offset,
    // which happens to be CX
    MOVQ X0, (CX)(FP)
    RET
