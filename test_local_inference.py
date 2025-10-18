#!/usr/bin/env python3
"""
Test script for local model inference with TransformersAdapter.

This script tests the TransformersAdapter with a simple task.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bench.adapters.transformers_adapter import TransformersAdapter
from bench.tools.tool_registry import ToolRegistry


def test_basic_inference():
    """Test basic inference without tools."""
    print("="*80)
    print("TEST 1: Basic Inference (No Tools)")
    print("="*80)
    
    # Use a small model for testing
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\nLoading model: {model_name}")
    print("Note: This will download the model if not cached locally")
    
    try:
        adapter = TransformersAdapter(
            model_name,
            device='auto',
            torch_dtype='auto',
            max_new_tokens=512,
            temperature=0.7,
        )
        
        messages = [
            {"role": "user", "content": "Hello! Can you tell me what is 2+2?"}
        ]
        
        print("\nGenerating response...")
        response = adapter.chat_completion(messages)
        
        print("\n" + "-"*80)
        print("Response:")
        print("-"*80)
        print(f"Content: {response['message']['content']}")
        print(f"\nTokens used:")
        print(f"  Input: {response['usage']['prompt_tokens']}")
        print(f"  Output: {response['usage']['completion_tokens']}")
        print(f"  Total: {response['usage']['total_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_calling():
    """Test inference with simulated tool calling."""
    print("\n\n" + "="*80)
    print("TEST 2: Tool Calling Simulation")
    print("="*80)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\nUsing model: {model_name}")
    
    try:
        adapter = TransformersAdapter(
            model_name,
            device='auto',
            torch_dtype='auto',
            max_new_tokens=512,
            temperature=0.7,
        )
        
        # Define a simple tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        messages = [
            {"role": "user", "content": "What's the weather like in Seoul?"}
        ]
        
        print("\nGenerating response with tools...")
        response = adapter.chat_completion(messages, tools=tools)
        
        print("\n" + "-"*80)
        print("Response:")
        print("-"*80)
        print(f"Content: {response['message']['content']}")
        
        if 'tool_calls' in response['message']:
            print(f"\nTool calls detected: {len(response['message']['tool_calls'])}")
            for i, tool_call in enumerate(response['message']['tool_calls'], 1):
                print(f"\nTool Call {i}:")
                print(f"  Name: {tool_call['function']['name']}")
                print(f"  Arguments: {tool_call['function']['arguments']}")
        else:
            print("\nNo tool calls detected (model may not support tool calling format)")
        
        print(f"\nTokens used:")
        print(f"  Input: {response['usage']['prompt_tokens']}")
        print(f"  Output: {response['usage']['completion_tokens']}")
        print(f"  Total: {response['usage']['total_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantized_inference():
    """Test inference with quantization."""
    print("\n\n" + "="*80)
    print("TEST 3: Quantized Inference (4-bit)")
    print("="*80)
    
    # Check if CUDA is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n[WARNING] CUDA not available, skipping quantization test")
            return True
    except ImportError:
        print("\n[WARNING] torch not available, skipping quantization test")
        return True
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\nLoading model with 4-bit quantization: {model_name}")
    
    try:
        adapter = TransformersAdapter(
            model_name,
            device='auto',
            torch_dtype='float16',
            quantization='4bit',
            max_new_tokens=256,
            temperature=0.7,
        )
        
        messages = [
            {"role": "user", "content": "What is Python programming language?"}
        ]
        
        print("\nGenerating response...")
        response = adapter.chat_completion(messages)
        
        print("\n" + "-"*80)
        print("Response:")
        print("-"*80)
        print(f"Content: {response['message']['content'][:200]}...")
        print(f"\nTokens: {response['usage']['total_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        print("Note: Quantization requires bitsandbytes library and CUDA")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TransformersAdapter Test Suite")
    print("="*60 + "\n")
    
    results = []
    
    # Test 1: Basic inference
    results.append(("Basic Inference", test_basic_inference()))
    
    # Test 2: Tool calling
    results.append(("Tool Calling", test_tool_calling()))
    
    # Test 3: Quantization (optional)
    results.append(("Quantization", test_quantized_inference()))
    
    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80)
    
    return all(passed for _, passed in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
