#!/usr/bin/env python3
"""
Quick example of using TransformersAdapter for local inference.

This demonstrates basic usage without running the full benchmark.
"""

from bench.adapters.transformers_adapter import TransformersAdapter

def main():
    print("="*80)
    print("TransformersAdapter - Quick Example")
    print("="*80)
    
    # Initialize adapter with a small model
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\nInitializing model: {model_name}")
    print("This will download the model if not cached (~1GB)")
    
    adapter = TransformersAdapter(
        model_name=model_name,
        device="auto",
        torch_dtype="auto",
        max_new_tokens=256,
        temperature=0.7,
    )
    
    print("\n✓ Model loaded successfully!")
    
    # Example 1: Simple question
    print("\n" + "-"*80)
    print("Example 1: Simple Question")
    print("-"*80)
    
    messages = [
        {"role": "user", "content": "What is the capital of South Korea?"}
    ]
    
    response = adapter.chat_completion(messages)
    print(f"\nQuestion: {messages[0]['content']}")
    print(f"Answer: {response['message']['content']}")
    
    # Example 2: Math problem
    print("\n" + "-"*80)
    print("Example 2: Math Problem")
    print("-"*80)
    
    messages = [
        {"role": "user", "content": "If I have 5 apples and buy 3 more, how many do I have?"}
    ]
    
    response = adapter.chat_completion(messages)
    print(f"\nQuestion: {messages[0]['content']}")
    print(f"Answer: {response['message']['content']}")
    
    # Example 3: With conversation history
    print("\n" + "-"*80)
    print("Example 3: Multi-turn Conversation")
    print("-"*80)
    
    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice! How can I help you today?"},
        {"role": "user", "content": "What's my name?"}
    ]
    
    response = adapter.chat_completion(messages)
    print(f"\nConversation:")
    for msg in messages:
        print(f"  {msg['role'].title()}: {msg['content']}")
    print(f"\nResponse: {response['message']['content']}")
    
    print("\n" + "="*80)
    print("Examples completed successfully!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
