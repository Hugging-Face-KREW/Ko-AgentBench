"""vLLM adapter for high-performance local model inference in Ko-AgentBench."""

import json
import re
import logging
from typing import Any, Dict, List, Optional
from .base_adapter import BaseAdapter
from ..observability import observe

try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    LLM = None
    SamplingParams = None
    GuidedDecodingParams = None


class VLLMAdapter(BaseAdapter):
    """vLLM adapter for high-performance local inference with tool calling support.
    
    vLLM provides optimized inference with:
    - PagedAttention for efficient memory management
    - Continuous batching for higher throughput
    - Optimized CUDA kernels
    - Support for quantization (AWQ, GPTQ, etc.)
    """
    
    def __init__(self, model_name: str, **config):
        """Initialize vLLM adapter with optimized settings.
        
        vLLM adapter only supports offline mode (local GPU inference).
        Models must be pre-downloaded to local cache.
        
        Args:
            model_name: Model identifier (with 'vllm/' prefix removed)
            **config: Configuration parameters
                - tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
                - gpu_memory_utilization: GPU memory utilization (0.0-1.0, default: 0.9)
                - max_model_len: Maximum model context length (default: None, auto-detect)
                - max_new_tokens: Maximum new tokens to generate (default: 1024)
                - temperature: Sampling temperature (default: 0.7)
                - top_p: Top-p sampling (default: 0.9)
                - quantization: Quantization method ('awq', 'gptq', 'squeezellm', None)
                - trust_remote_code: Whether to trust remote code (default: True)
                - dtype: Data type ('auto', 'float16', 'bfloat16', 'float32')
        """
        if LLM is None or SamplingParams is None:
            raise ImportError(
                "vllm package is required for VLLMAdapter. "
                "Install it with: pip install vllm"
            )
        
        # Remove 'vllm/' prefix if present
        if model_name.startswith('vllm/'):
            model_name = model_name[5:]
        
        super().__init__(model_name, **config)
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        self.tensor_parallel_size = config.get('tensor_parallel_size', 1)
        self.gpu_memory_utilization = config.get('gpu_memory_utilization', 0.9)
        self.max_model_len = config.get('max_model_len', None)
        self.max_new_tokens = config.get('max_new_tokens', 1024)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 0.9)
        self.quantization = config.get('quantization', None)
        self.trust_remote_code = config.get('trust_remote_code', True)
        self.dtype = config.get('dtype', 'auto')
        
        # Load model
        self.logger.info(f"Loading model with vLLM: {model_name}")
        self._load_model()
        self.logger.info(f"vLLM model loaded successfully")
        
        # Setup context management
        self._setup_context_management()
    
    def _load_model(self):
        """Load model using vLLM with optimized settings."""
        vllm_kwargs = {
            'model': self.model_name,
            'tensor_parallel_size': self.tensor_parallel_size,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'trust_remote_code': self.trust_remote_code,
            'dtype': self.dtype,
            # Disable compilation features to avoid Python.h dependency issues
            'enforce_eager': True,  # Disable CUDA graphs
            'disable_custom_all_reduce': True,  # Disable custom kernels
        }
        
        # Add max_model_len if specified
        if self.max_model_len:
            vllm_kwargs['max_model_len'] = self.max_model_len
        
        # Add quantization if specified
        if self.quantization:
            vllm_kwargs['quantization'] = self.quantization
            self.logger.info(f"Using {self.quantization} quantization")
        
        self.llm = LLM(**vllm_kwargs)
        
        # Get tokenizer from vLLM
        self.tokenizer = self.llm.get_tokenizer()
    
    def _setup_context_management(self):
        """Setup context window management based on model config."""
        # Get max model length from vLLM
        if hasattr(self.llm.llm_engine, 'model_config'):
            model_config = self.llm.llm_engine.model_config
            model_max_length = getattr(model_config, 'max_model_len', None)
            
            if model_max_length:
                self.logger.info(f"[CONFIG] Detected max_model_len: {model_max_length}")
            else:
                model_max_length = 8192  # Conservative default
                self.logger.warning(f"[CONFIG] Could not detect max_model_len, using default: {model_max_length}")
        else:
            model_max_length = 8192
            self.logger.warning(f"[CONFIG] Could not access model_config, using default: {model_max_length}")
        
        # Calculate max_context_tokens
        buffer = 100
        self.max_context_tokens = max(512, model_max_length - self.max_new_tokens - buffer)
        
        self.logger.info(
            f"[CONFIG] Context management configured: "
            f"max_context_tokens={self.max_context_tokens}, "
            f"max_new_tokens={self.max_new_tokens}"
        )
    
    @observe(as_type="generation")
    def chat_completion(self, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict]] = None,
                       **kwargs) -> Dict[str, Any]:
        """Generate chat completion using vLLM.
        
        Args:
            messages: List of chat messages
            tools: Available tools for function calling
            **kwargs: Additional generation parameters (including 'n' for multiple completions)
            
        Returns:
            Chat completion response in canonical format
        """
        # Check if multiple completions requested
        n = kwargs.pop('n', 1)
        
        if n > 1:
            # Generate multiple completions using vLLM's native support
            self.logger.info(f"[VLLM] Generating {n} completions")
            return self._multi_chat_completion(messages, tools, n, **kwargs)
        else:
            # Single completion
            return self._single_chat_completion(messages, tools, **kwargs)
    
    def _single_chat_completion(self, messages: List[Dict[str, str]], 
                                tools: Optional[List[Dict]] = None,
                                **kwargs) -> Dict[str, Any]:
        """Generate a single chat completion.
        
        Args:
            messages: List of chat messages
            tools: Available tools for function calling
            **kwargs: Additional generation parameters
            
        Returns:
            Chat completion response in canonical format
        """
        self.logger.info(f"[VLLM DEBUG] chat_completion called")
        self.logger.info(f"[VLLM DEBUG] Messages count: {len(messages)}")
        self.logger.info(f"[VLLM DEBUG] Tools count: {len(tools) if tools else 0}")
        
        # Truncate messages if needed
        messages = self._truncate_messages_if_needed(messages, tools)
        
        # Apply chat template
        prompt = self._apply_chat_template(messages, tools)
        
        # Ensure prompt is within limit
        prompt = self._ensure_prompt_within_limit(prompt)
        
        self.logger.info(f"[VLLM DEBUG] Final prompt length: {len(prompt)} chars")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
            temperature=kwargs.get('temperature', self.temperature),
            top_p=kwargs.get('top_p', self.top_p),
        )
        
        # Generate
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        # Extract generated text
        generated_text = output.outputs[0].text
        
        self.logger.info(f"[VLLM DEBUG] Generated text length: {len(generated_text)} chars")
        self.logger.info(f"[VLLM DEBUG] Generated text:\n{generated_text}")
        
        # Calculate token usage
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        
        # Parse tool calls if tools were provided
        tool_calls = None
        response_content = generated_text
        
        if tools:
            self.logger.info(f"[VLLM PARSE] Attempting to parse tool calls")
            tool_calls, response_content = self._parse_tool_calls(generated_text, tools)
            if tool_calls:
                self.logger.info(f"[VLLM PARSE] Successfully parsed {len(tool_calls)} tool call(s)")
            else:
                self.logger.info(f"[VLLM PARSE] No tool calls found")
        
        # Build canonical response
        response = {
            "message": {
                "role": "assistant",
                "content": response_content
            },
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "model": self.model_name,
            "finish_reason": output.outputs[0].finish_reason or "stop"
        }
        
        if tool_calls:
            response["message"]["tool_calls"] = tool_calls
            response["finish_reason"] = "tool_calls"
        
        return response
    
    def _multi_chat_completion(self, messages: List[Dict[str, str]], 
                               tools: Optional[List[Dict]] = None,
                               n: int = 1,
                               **kwargs) -> Dict[str, Any]:
        """Generate multiple chat completions efficiently using vLLM.
        
        Args:
            messages: List of chat messages
            tools: Available tools for function calling
            n: Number of completions to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Dict with 'choices' containing list of completion results
        """
        self.logger.info(f"[VLLM] Generating {n} completions")
        
        # Prepare prompt
        messages = self._truncate_messages_if_needed(messages, tools)
        prompt = self._apply_chat_template(messages, tools)
        prompt = self._ensure_prompt_within_limit(prompt)
        
        # Create sampling parameters with n > 1
        sampling_params = SamplingParams(
            n=n,
            max_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
            temperature=kwargs.get('temperature', self.temperature),
            top_p=kwargs.get('top_p', self.top_p),
        )
        
        # Generate all completions in one call (vLLM handles this efficiently)
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        # Process each completion
        results = []
        for completion_output in output.outputs:
            generated_text = completion_output.text
            
            # Parse tool calls if tools were provided
            tool_calls = None
            response_content = generated_text
            
            if tools:
                tool_calls, response_content = self._parse_tool_calls(generated_text, tools)
            
            # Build response for this completion
            result = {
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "usage": {
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(completion_output.token_ids),
                    "total_tokens": len(output.prompt_token_ids) + len(completion_output.token_ids)
                },
                "model": self.model_name,
                "finish_reason": completion_output.finish_reason or "stop"
            }
            
            if tool_calls:
                result["message"]["tool_calls"] = tool_calls
                result["finish_reason"] = "tool_calls"
            
            results.append(result)
        
        return {"choices": results, "n": n}
    
    def _apply_chat_template(self, messages: List[Dict[str, str]], 
                            tools: Optional[List[Dict]] = None) -> str:
        """Apply chat template to messages and tools.
        
        Args:
            messages: List of chat messages
            tools: Available tools
            
        Returns:
            Formatted prompt string
        """
        self.logger.info(f"[VLLM TEMPLATE] Applying chat template with {len(tools) if tools else 0} tools")
        
        # Try to use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                formatted_messages = []
                for msg in messages:
                    role = msg.get('role')
                    content = msg.get('content', '')
                    
                    if role == 'tool':
                        formatted_messages.append({
                            'role': 'tool',
                            'content': content,
                            'tool_call_id': msg.get('tool_call_id')
                        })
                    else:
                        formatted_messages.append({'role': role, 'content': content})
                
                template_kwargs = {'add_generation_prompt': True}
                
                if tools:
                    try:
                        template_kwargs['tools'] = tools
                        prompt = self.tokenizer.apply_chat_template(
                            formatted_messages,
                            tokenize=False,
                            **template_kwargs
                        )
                        self.logger.info(f"[VLLM TEMPLATE] Successfully applied template with tools")
                        return prompt
                    except Exception as e:
                        self.logger.warning(f"[VLLM TEMPLATE] Failed to pass tools to template: {e}")
                        # Add tools to system message
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
        
        # Fallback to manual formatting
        return self._manual_chat_format(messages, tools)
    
    def _format_tools_as_text(self, tools: List[Dict]) -> str:
        """Format tools as text description."""
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
        """Manually format messages into a prompt string."""
        prompt = ""
        
        if tools:
            prompt += self._format_tools_as_text(tools) + "\n\n"
        
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
    
    def _truncate_messages_if_needed(self, messages: List[Dict[str, str]], 
                                     tools: Optional[List[Dict]] = None) -> List[Dict[str, str]]:
        """Truncate messages to prevent context overflow."""
        if not messages:
            return messages
        
        messages = [msg.copy() for msg in messages]
        
        # Rough estimate
        total_chars = sum(len(msg.get('content', '')) for msg in messages)
        
        if total_chars // 4 < self.max_context_tokens * 0.8:
            return messages
        
        # Remove older messages
        system_msg = None
        if messages and messages[0].get('role') == 'system':
            system_msg = messages.pop(0)
        
        # Keep last user message and subsequent messages
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get('role') == 'user':
                last_user_idx = i
                break
        
        if last_user_idx > 0:
            keep_from = max(0, last_user_idx - 5)
            messages = messages[keep_from:]
            self.logger.warning(f"[VLLM CONTEXT] Removed {keep_from} older messages (messages before index {keep_from})")
        
        if system_msg:
            messages.insert(0, system_msg)
        
        return messages
    
    def _ensure_prompt_within_limit(self, prompt: str) -> str:
        """Ensure final prompt is within token limit."""
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        if len(tokens) <= self.max_context_tokens:
            return prompt
        
        self.logger.warning(
            f"[VLLM CONTEXT] Prompt exceeds limit: {len(tokens)} > {self.max_context_tokens}, truncating..."
        )
        
        # Keep first 30% and last 60%
        keep_start = int(self.max_context_tokens * 0.3)
        keep_end = int(self.max_context_tokens * 0.6)
        
        ellipsis_token = self.tokenizer.encode("...", add_special_tokens=False)
        truncated_tokens = tokens[:keep_start] + ellipsis_token + tokens[-keep_end:]
        
        if len(truncated_tokens) > self.max_context_tokens:
            truncated_tokens = truncated_tokens[:self.max_context_tokens]
        
        truncated_prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        self.logger.warning(
            f"[VLLM CONTEXT] Truncated to {len(truncated_tokens)} tokens"
        )
        
        return truncated_prompt
    
    def _parse_tool_calls(self, text: str, tools: List[Dict]) -> tuple[Optional[List[Dict]], str]:
        """Parse tool calls from model output.
        
        Supports multiple formats (same as TransformersAdapter).
        """
        tool_calls = []
        remaining_text = text
        
        # Same patterns as TransformersAdapter
        xml_pattern = r'<tool_call>\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})\s*</tool_call>'
        tools_tag_pattern = r'<tools>\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})\s*</tools>'
        function_tag_pattern = r'<function=([^>]+)>(\{[^<]+\})</function>'
        single_tool_name_pattern = r'\{"tool_name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]+\})\}'
        single_name_pattern = r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]+\})\}'
        array_pattern = r'\[\s*\{[^\]]+\}\s*\]'
        
        try:
            # Try <tools> tag format
            tools_tag_matches = re.finditer(tools_tag_pattern, text, re.DOTALL)
            tools_tag_found = False
            
            for match in tools_tag_matches:
                tools_tag_found = True
                json_str = match.group(1)
                try:
                    call_data = json.loads(json_str)
                    
                    if 'function' in call_data:
                        func_data = call_data['function']
                        tool_name = func_data.get('name')
                        arguments = func_data.get('parameters', {})
                        if 'properties' in arguments:
                            arguments = arguments.get('properties', {})
                    else:
                        tool_name = call_data.get('tool_name') or call_data.get('name')
                        arguments = call_data.get('arguments', {})
                    
                    if tool_name:
                        tool_calls.append({
                            'id': f'call_{len(tool_calls)}',
                            'type': 'function',
                            'function': {
                                'name': tool_name,
                                'arguments': json.dumps(arguments) if isinstance(arguments, dict) else arguments
                            }
                        })
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse <tools> tag JSON: {e}")
                    continue
            
            if tools_tag_found:
                remaining_text = re.sub(tools_tag_pattern, '', text, flags=re.DOTALL).strip()
                return (tool_calls if tool_calls else None), remaining_text
            
            # Try function tag format
            function_matches = re.finditer(function_tag_pattern, text, re.DOTALL)
            function_found = False
            
            for match in function_matches:
                function_found = True
                tool_name = match.group(1).strip()
                arguments_str = match.group(2).strip()
                
                try:
                    arguments = json.loads(arguments_str)
                    tool_calls.append({
                        'id': f'call_{len(tool_calls)}',
                        'type': 'function',
                        'function': {
                            'name': tool_name,
                            'arguments': json.dumps(arguments) if isinstance(arguments, dict) else arguments
                        }
                    })
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse function tag: {e}")
                    continue
            
            if function_found:
                remaining_text = re.sub(function_tag_pattern, '', text, flags=re.DOTALL).strip()
                return (tool_calls if tool_calls else None), remaining_text
            
            # Try XML pattern
            xml_matches = re.finditer(xml_pattern, text, re.DOTALL)
            xml_found = False
            
            for match in xml_matches:
                xml_found = True
                json_str = match.group(1)
                try:
                    call_data = json.loads(json_str)
                    tool_name = call_data.get('tool_name') or call_data.get('name')
                    arguments = call_data.get('arguments', {})
                    
                    if tool_name:
                        tool_calls.append({
                            'id': f'call_{len(tool_calls)}',
                            'type': 'function',
                            'function': {
                                'name': tool_name,
                                'arguments': json.dumps(arguments) if isinstance(arguments, dict) else arguments
                            }
                        })
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse XML tool call: {e}")
                    continue
            
            if xml_found:
                remaining_text = re.sub(xml_pattern, '', text, flags=re.DOTALL).strip()
                return (tool_calls if tool_calls else None), remaining_text
            
            # Try array pattern
            array_match = re.search(array_pattern, text, re.DOTALL)
            if array_match:
                json_str = array_match.group(0)
                calls_data = json.loads(json_str)
                
                for call_data in calls_data:
                    tool_name = call_data.get('tool_name') or call_data.get('name')
                    arguments = call_data.get('arguments', {})
                    
                    if tool_name:
                        tool_calls.append({
                            'id': f'call_{len(tool_calls)}',
                            'type': 'function',
                            'function': {
                                'name': tool_name,
                                'arguments': json.dumps(arguments) if isinstance(arguments, dict) else arguments
                            }
                        })
                
                remaining_text = text[:array_match.start()] + text[array_match.end():]
                return (tool_calls if tool_calls else None), remaining_text.strip()
            
            # Try single pattern with tool_name
            single_match = re.search(single_tool_name_pattern, text)
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
                
                remaining_text = text[:single_match.start()] + text[single_match.end():]
                return (tool_calls if tool_calls else None), remaining_text.strip()
            
            # Try single pattern with name
            single_match = re.search(single_name_pattern, text)
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
                
                remaining_text = text[:single_match.start()] + text[single_match.end():]
                return (tool_calls if tool_calls else None), remaining_text.strip()
        
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse tool calls: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing tool calls: {e}")
        
        return (tool_calls if tool_calls else None), remaining_text.strip()
    
    def convert_to_provider_format(self, messages: List[Dict[str, str]], 
                                  tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Convert to vLLM format (for compatibility)."""
        return {
            "messages": messages,
            "tools": tools
        }
    
    def convert_from_provider_format(self, response: Any) -> Dict[str, Any]:
        """Convert vLLM output to canonical format (handled in chat_completion)."""
        return response
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'llm'):
            del self.llm
