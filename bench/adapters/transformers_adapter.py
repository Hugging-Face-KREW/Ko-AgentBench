"""Transformers adapter for local model inference in Ko-AgentBench."""

import json
import re
import logging
from typing import Any, Dict, List, Optional, Union
from .base_adapter import BaseAdapter
from ..observability import observe

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None


class TransformersAdapter(BaseAdapter):
    """Transformers adapter for local model inference with tool calling support."""
    
    def __init__(self, model_name: str, **config):
        """Initialize Transformers adapter with local model loading.
        
        Args:
            model_name: HuggingFace model ID (e.g., 'Qwen/Qwen2.5-7B-Instruct')
            **config: Configuration parameters
                - device: Device to load model on ('cuda', 'cpu', 'auto')
                - device_map: Device mapping strategy ('auto', 'balanced', etc.)
                - torch_dtype: Torch data type ('float16', 'bfloat16', 'float32', 'auto')
                - quantization: Quantization config ('4bit', '8bit', None)
                - max_new_tokens: Maximum new tokens to generate (default: 1024)
                - temperature: Sampling temperature (default: 0.7)
                - top_p: Top-p sampling (default: 0.9)
                - trust_remote_code: Whether to trust remote code (default: False)
        """
        if torch is None or AutoModelForCausalLM is None:
            raise ImportError(
                "transformers and torch packages are required for TransformersAdapter. "
                "Install them with: pip install transformers torch accelerate"
            )
            
        super().__init__(model_name, **config)
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        self.device = config.get('device', 'auto')
        self.device_map = config.get('device_map', 'auto')
        self.torch_dtype = self._get_torch_dtype(config.get('torch_dtype', 'auto'))
        self.quantization = config.get('quantization', None)
        self.max_new_tokens = config.get('max_new_tokens', 1024)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 0.9)
        self.trust_remote_code = config.get('trust_remote_code', False)
        
        # Load model and tokenizer
        self.logger.info(f"Loading model: {model_name}")
        self._load_model_and_tokenizer()
        self.logger.info(f"Model loaded successfully on {self.device}")
    
    def _get_torch_dtype(self, dtype_str: str):
        """Convert dtype string to torch dtype."""
        if dtype_str == 'auto':
            return 'auto'
        elif dtype_str == 'float16' or dtype_str == 'fp16':
            return torch.float16
        elif dtype_str == 'bfloat16' or dtype_str == 'bf16':
            return torch.bfloat16
        elif dtype_str == 'float32' or dtype_str == 'fp32':
            return torch.float32
        else:
            return 'auto'
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with specified configuration."""
        # Prepare model loading kwargs
        model_kwargs = {
            'device_map': self.device_map,
            'torch_dtype': self.torch_dtype,
            'trust_remote_code': self.trust_remote_code,
        }
        
        # Add quantization config if specified
        if self.quantization == '4bit':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs['quantization_config'] = quantization_config
            self.logger.info("Using 4-bit quantization")
        elif self.quantization == '8bit':
            model_kwargs['load_in_8bit'] = True
            self.logger.info("Using 8-bit quantization")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @observe(as_type="generation")
    def chat_completion(self, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict]] = None,
                       **kwargs) -> Dict[str, Any]:
        """Generate chat completion using local transformers model.
        
        Args:
            messages: List of chat messages
            tools: Available tools for function calling
            **kwargs: Additional generation parameters
            
        Returns:
            Chat completion response in canonical format
        """
        # Apply chat template
        prompt = self._apply_chat_template(messages, tools)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move to device
        if self.device != 'auto':
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # For auto device_map, inputs should go to the first device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        generation_config = {
            'max_new_tokens': kwargs.get('max_new_tokens', self.max_new_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'top_p': kwargs.get('top_p', self.top_p),
            'do_sample': kwargs.get('temperature', self.temperature) > 0,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Calculate token usage (approximate)
        input_tokens = inputs['input_ids'].shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        
        # Parse tool calls if tools were provided
        tool_calls = None
        response_content = generated_text
        
        if tools:
            tool_calls, response_content = self._parse_tool_calls(generated_text, tools)
        
        # Build canonical response
        response = {
            "message": {
                "role": "assistant",
                "content": response_content
            },
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "model": self.model_name,
            "finish_reason": "stop"
        }
        
        if tool_calls:
            response["message"]["tool_calls"] = tool_calls
            response["finish_reason"] = "tool_calls"
        
        return response
    
    def _apply_chat_template(self, messages: List[Dict[str, str]], 
                            tools: Optional[List[Dict]] = None) -> str:
        """Apply chat template to messages and tools.
        
        Args:
            messages: List of chat messages
            tools: Available tools
            
        Returns:
            Formatted prompt string
        """
        # Try to use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                # Convert messages to the format expected by chat template
                formatted_messages = []
                for msg in messages:
                    role = msg.get('role')
                    content = msg.get('content', '')
                    
                    # Handle tool results
                    if role == 'tool':
                        # Some models expect tool results in a specific format
                        formatted_messages.append({
                            'role': 'tool',
                            'content': content,
                            'tool_call_id': msg.get('tool_call_id')
                        })
                    else:
                        formatted_messages.append({'role': role, 'content': content})
                
                # Add tools to the template if supported
                template_kwargs = {'add_generation_prompt': True}
                if tools:
                    # Try to add tools (some models support this)
                    try:
                        template_kwargs['tools'] = tools
                    except:
                        # If tools not supported in template, add to system message
                        tools_description = self._format_tools_as_text(tools)
                        if formatted_messages and formatted_messages[0]['role'] == 'system':
                            formatted_messages[0]['content'] += f"\n\n{tools_description}"
                        else:
                            formatted_messages.insert(0, {
                                'role': 'system',
                                'content': tools_description
                            })
                
                prompt = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    **template_kwargs
                )
                return prompt
            except Exception as e:
                self.logger.warning(f"Chat template failed: {e}, falling back to manual formatting")
        
        # Fallback: Manual formatting
        return self._manual_chat_format(messages, tools)
    
    def _format_tools_as_text(self, tools: List[Dict]) -> str:
        """Format tools as text description for models without native tool support.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            Text description of available tools
        """
        tools_text = "You have access to the following tools:\n\n"
        
        for tool in tools:
            func = tool.get('function', {})
            name = func.get('name', 'unknown')
            description = func.get('description', '')
            parameters = func.get('parameters', {})
            
            tools_text += f"Tool: {name}\n"
            tools_text += f"Description: {description}\n"
            
            if parameters and 'properties' in parameters:
                tools_text += "Parameters:\n"
                for param_name, param_info in parameters['properties'].items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    required = param_name in parameters.get('required', [])
                    tools_text += f"  - {param_name} ({param_type}){' [required]' if required else ''}: {param_desc}\n"
            
            tools_text += "\n"
        
        tools_text += (
            "To use a tool, respond with a JSON object in this format:\n"
            '{"tool_name": "<tool_name>", "arguments": {"param1": "value1", "param2": "value2"}}\n\n'
            "You can call multiple tools by providing a JSON array of tool calls."
        )
        
        return tools_text
    
    def _manual_chat_format(self, messages: List[Dict[str, str]], 
                           tools: Optional[List[Dict]] = None) -> str:
        """Manually format messages into a prompt string.
        
        Args:
            messages: List of chat messages
            tools: Available tools
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        
        # Add tools description if provided
        if tools:
            prompt += self._format_tools_as_text(tools) + "\n\n"
        
        # Format messages
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
            elif role == 'tool':
                tool_call_id = msg.get('tool_call_id', 'unknown')
                prompt += f"Tool Result [{tool_call_id}]: {content}\n\n"
        
        prompt += "Assistant: "
        return prompt
    
    def _parse_tool_calls(self, text: str, tools: List[Dict]) -> tuple[Optional[List[Dict]], str]:
        """Parse tool calls from model output.
        
        Args:
            text: Generated text from model
            tools: Available tools
            
        Returns:
            Tuple of (tool_calls, remaining_text)
        """
        tool_calls = []
        remaining_text = text
        
        # Try to find JSON objects in the text
        # Pattern 1: Single tool call
        single_pattern = r'\{"tool_name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]+\})\}'
        
        # Pattern 2: Array of tool calls
        array_pattern = r'\[\s*\{[^\]]+\}\s*\]'
        
        try:
            # Try array pattern first
            array_match = re.search(array_pattern, text, re.DOTALL)
            if array_match:
                json_str = array_match.group(0)
                calls_data = json.loads(json_str)
                
                for i, call_data in enumerate(calls_data):
                    tool_name = call_data.get('tool_name')
                    arguments = call_data.get('arguments', {})
                    
                    if tool_name:
                        tool_calls.append({
                            'id': f'call_{i}',
                            'type': 'function',
                            'function': {
                                'name': tool_name,
                                'arguments': json.dumps(arguments)
                            }
                        })
                
                # Remove the JSON from text
                remaining_text = text[:array_match.start()] + text[array_match.end():]
            
            # Try single pattern
            else:
                single_match = re.search(single_pattern, text)
                if single_match:
                    tool_name = single_match.group(1)
                    arguments_str = single_match.group(2)
                    
                    tool_calls.append({
                        'id': 'call_0',
                        'type': 'function',
                        'function': {
                            'name': tool_name,
                            'arguments': arguments_str
                        }
                    })
                    
                    # Remove the JSON from text
                    remaining_text = text[:single_match.start()] + text[single_match.end():]
        
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse tool calls: {e}")
        
        return (tool_calls if tool_calls else None), remaining_text.strip()
    
    def convert_to_provider_format(self, messages: List[Dict[str, str]], 
                                  tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Convert to transformers format (not used directly, for compatibility).
        
        Args:
            messages: Messages in canonical format
            tools: Tools in canonical format
            
        Returns:
            Dict with messages and tools
        """
        return {
            "messages": messages,
            "tools": tools
        }
    
    def convert_from_provider_format(self, response: Any) -> Dict[str, Any]:
        """Convert transformers output to canonical format (handled in chat_completion).
        
        Args:
            response: Model output
            
        Returns:
            Canonical format response
        """
        return response
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        # Clear CUDA cache if available
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
