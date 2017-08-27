package main

import (
	"fmt"
	"log"

	"github.com/xfong/go2opencl/cl"
)

var _kernelSource = `
__kernel void check(
   __global float* value,
   __global int* lastColIndex
   __global float*[] splitGroup,
   const unsigned int total,
   unsigned int count) {
   int i = get_global_id(0);
   if (i < total) {

   }
}
`

var _context *cl.Context
var _queue *cl.CommandQueue
var _program *cl.Program

func withValueGPU(lastColIndex int, value float32, splitGroup []datarow) (count float32) {
	// first time, setup context
	if _context == nil {
		var err error
		fmt.Println("setting up compute context...")
		_context, err = cl.CreateContext([]*cl.Device{processor})
		if err != nil {
			log.Fatalf("CreateContext failed: %+v", err)
		}

		_queue, err = _context.CreateCommandQueue(processor, 0)
		if err != nil {
			log.Fatalf("CreateCommandQueue failed: %+v", err)
		}

		_program, err = _context.CreateProgramWithSource([]string{_kernelSource})
		if err != nil {
			log.Fatalf("CreateProgramWithSource failed: %+v", err)
		}
		if err := _program.BuildProgram(nil, ""); err != nil {
			log.Fatalf("BuildProgram failed: %+v", err)
		}

		kernel, err := _program.CreateKernel("square")
		if err != nil {
			log.Fatalf("CreateKernel failed: %+v", err)
		}

		totalArgs, err := kernel.NumArgs()
		if err != nil {
			log.Fatalf("Failed to get number of arguments of kernel: %+v", err)
		}
		fmt.Printf("Number of arguments in kernel : %d", totalArgs)
		fmt.Println("Done setting up context.")
	}
	// end context setup

	input, err := _context.CreateEmptyBuffer(cl.MemReadOnly, 4*len(data))
	if err != nil {
		log.Fatalf("CreateBuffer failed for input: %+v", err)
	}

	output, err := context.CreateEmptyBuffer(MemReadOnly, 4*len(data))
	if err != nil {
		log.Fatalf("CreateBuffer failed for output: %+v", err)
	}

	if _, err := queue.EnqueueWriteBufferFloat32(input, true, 0, data[:], nil); err != nil {
		log.Fatalf("EnqueueWriteBufferFloat32 failed: %+v", err)
	}

	if err := kernel.SetArgs(input, output, uint32(len(data))); err != nil {
		log.Fatalf("SetKernelArgs failed: %+v", err)
	}

	local, err := kernel.WorkGroupSize(device)
	if err != nil {
		log.Fatalf("WorkGroupSize failed: %+v", err)
	}
	t.Logf("Work group size: %d", local)

	global := len(data)
	d := len(data) % local
	if d != 0 {
		global += local - d
	}
	t.Logf("Global work group size: %d ", global)
	if _, err := queue.EnqueueNDRangeKernel(kernel, nil, []int{global}, []int{local}, nil); err != nil {
		log.Fatalf("EnqueueNDRangeKernel failed: %+v", err)
	}

	if err := queue.Finish(); err != nil {
		log.Fatalf("Finish failed: %+v", err)
	}

	results := make([]float32, len(data))
	if _, err := queue.EnqueueReadBufferFloat32(output, true, 0, results, nil); err != nil {
		log.Fatalf("EnqueueReadBufferFloat32 failed: %+v", err)
	}

	//correct := 0
	//for i, v := range data {
	//	if results[i] == v*v {
	//		correct++
	//	}
	//}
	//
	//if correct != len(data) {
	//	t.Error("%d/%d correct values", correct, len(results))
	//}

	splitGroupLen := len(splitGroup)
	var prediction float32
	for i := 0; i < splitGroupLen; i++ {
		prediction = splitGroup[i][lastColIndex]
		if prediction == value {
			count++
		}
	}

	return count
}

// this function takes up about 91 - 98% of cpu burn.
func withValue(lastColIndex int, value float32, splitGroup []datarow) (count float32) {
	splitGroupLen := len(splitGroup)
	var prediction float32
	for i := 0; i < splitGroupLen; i++ {
		prediction = splitGroup[i][lastColIndex]
		if prediction == value {
			count++
		}
	}

	return count
}
