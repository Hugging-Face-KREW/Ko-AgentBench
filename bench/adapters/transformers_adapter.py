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
                - dtype: Torch data type ('float16', 'bfloat16', 'float32', 'auto')
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
        # Support both 'dtype' (new) and 'torch_dtype' (old) for backward compatibility
        dtype_value = config.get('dtype') or config.get('torch_dtype', 'auto')
        self.dtype = self._get_torch_dtype(dtype_value)
        self.quantization = config.get('quantization', None)
        self.max_new_tokens = config.get('max_new_tokens', 1024)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 0.9)
        self.trust_remote_code = config.get('trust_remote_code', True)
        
        # Load model and tokenizer first to get model config
        self.logger.info(f"Loading model: {model_name}")
        self._load_model_and_tokenizer()
        self.logger.info(f"Model loaded successfully on {self.device}")
        
        # Context window management - auto-detect from model config
        self._setup_context_management(config)
    
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
            'torch_dtype': self.dtype,
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
        
        # DEBUG: Log chat template info
        if hasattr(self.tokenizer, 'chat_template'):
            template = self.tokenizer.chat_template
            if template:
                self.logger.info(f"[INIT DEBUG] Model has chat_template")
                self.logger.info(f"[INIT DEBUG] Template preview (first 500 chars):\n{str(template)[:500]}")
            else:
                self.logger.warning(f"[INIT DEBUG] Model has chat_template attribute but it's None/empty")
        else:
            self.logger.warning(f"[INIT DEBUG] Model does NOT have chat_template attribute")
    
    def _setup_context_management(self, config: Dict[str, Any]):
        """Setup context window management based on model config.
        
        Auto-detects max_position_embeddings from model config and sets
        appropriate limits for context and tool results.
        
        Args:
            config: User-provided configuration (can override auto-detection)
        """
        # Try to get model's maximum context length from various sources
        model_max_length = None
        
        # Method 1: Check model.config.max_position_embeddings
        if hasattr(self.model, 'config'):
            model_max_length = getattr(self.model.config, 'max_position_embeddings', None)
            if model_max_length:
                print(f"[CONTEXT CONFIG] Detected max_position_embeddings: {model_max_length}")
                self.logger.info(f"[CONFIG] Detected max_position_embeddings: {model_max_length}")
        
        # Method 2: Check tokenizer.model_max_length
        if not model_max_length and hasattr(self.tokenizer, 'model_max_length'):
            tokenizer_max = self.tokenizer.model_max_length
            # Some tokenizers return very large numbers (1e30) as default
            if tokenizer_max and tokenizer_max < 1000000:
                model_max_length = tokenizer_max
                print(f"[CONTEXT CONFIG] Using tokenizer.model_max_length: {model_max_length}")
                self.logger.info(f"[CONFIG] Using tokenizer.model_max_length: {model_max_length}")
        
        # Method 3: Check model.config.n_positions (GPT-style)
        if not model_max_length and hasattr(self.model, 'config'):
            n_positions = getattr(self.model.config, 'n_positions', None)
            if n_positions:
                model_max_length = n_positions
                self.logger.info(f"[CONFIG] Using n_positions: {model_max_length}")
        
        # Fallback: Use conservative default
        if not model_max_length:
            model_max_length = 8192  # Conservative default for modern models
            print(f"[CONTEXT CONFIG] Could not detect model max length, using default: {model_max_length}")
            self.logger.warning(
                f"[CONFIG] Could not detect model max length, using default: {model_max_length}"
            )
        
        # Calculate max_context_tokens: reserve space for generation
        # User can override via config
        if 'max_context_tokens' in config:
            self.max_context_tokens = config['max_context_tokens']
            print(f"[CONTEXT CONFIG] Using user-specified max_context_tokens: {self.max_context_tokens}")
            self.logger.info(f"[CONFIG] Using user-specified max_context_tokens: {self.max_context_tokens}")
        else:
            # Auto-calculate: model_max - max_new_tokens - small buffer
            buffer = 100  # Small buffer for safety
            self.max_context_tokens = max(512, model_max_length - self.max_new_tokens - buffer)
            print(
                f"[CONTEXT CONFIG] Auto-calculated max_context_tokens: {self.max_context_tokens} "
                f"(model_max={model_max_length}, max_new_tokens={self.max_new_tokens}, buffer={buffer})"
            )
            self.logger.info(
                f"[CONFIG] Auto-calculated max_context_tokens: {self.max_context_tokens} "
                f"(model_max={model_max_length}, max_new_tokens={self.max_new_tokens}, buffer={buffer})"
            )
        
        print(
            f"[CONTEXT CONFIG] ✓ Context management configured: "
            f"max_context_tokens={self.max_context_tokens}"
        )
        self.logger.info(
            f"[CONFIG] Context management configured: "
            f"max_context_tokens={self.max_context_tokens}"
        )
    
    @observe(as_type="generation")
    def chat_completion(self, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict]] = None,
                       **kwargs) -> Dict[str, Any]:
        """Generate chat completion using local transformers model.
        
        Args:
            messages: List of chat messages
            tools: Available tools for function calling
            **kwargs: Additional generation parameters (including 'n' for multiple completions)
            
        Returns:
            Chat completion response in canonical format (or dict with 'choices' if n > 1)
        """
        # Check if multiple completions requested
        n = kwargs.pop('n', 1)
        
        if n > 1:
            # Generate multiple completions by running inference multiple times
            self.logger.info(f"[REPETITION] Generating {n} completions")
            results = []
            
            for i in range(n):
                self.logger.info(f"[REPETITION] Generating completion {i+1}/{n}")
                result = self._single_chat_completion(messages, tools, **kwargs)
                results.append(result)
            
            return {"choices": results, "n": n}
        else:
            # Single completion (default behavior)
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
        # DEBUG: Log input
        self.logger.info(f"[CHAT DEBUG] chat_completion called")
        self.logger.info(f"[CHAT DEBUG] Messages count: {len(messages)}")
        self.logger.info(f"[CHAT DEBUG] Tools count: {len(tools) if tools else 0}")
        
        # Truncate messages if needed before applying template
        messages = self._truncate_messages_if_needed(messages, tools)
        
        # Apply chat template
        prompt = self._apply_chat_template(messages, tools)
        
        # Final check and truncate prompt if still too long
        prompt = self._ensure_prompt_within_limit(prompt)
        
        self.logger.info(f"[CHAT DEBUG] Final prompt length: {len(prompt)} chars")
        self.logger.info(f"[CHAT DEBUG] Prompt preview (last 800 chars):\n{prompt[-800:]}")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Remove token_type_ids if present (not used by decoder-only models)
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        
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
        
        self.logger.info(f"[CHAT DEBUG] Generated text length: {len(generated_text)} chars")
        self.logger.info(f"[CHAT DEBUG] Generated text:\n{generated_text}")
        
        # Calculate token usage (approximate)
        input_tokens = inputs['input_ids'].shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        
        # Parse tool calls if tools were provided
        tool_calls = None
        response_content = generated_text
        
        if tools:
            self.logger.info(f"[PARSE DEBUG] Attempting to parse tool calls from generated text")
            tool_calls, response_content = self._parse_tool_calls(generated_text, tools)
            if tool_calls:
                self.logger.info(f"[PARSE DEBUG] Successfully parsed {len(tool_calls)} tool call(s)")
                for tc in tool_calls:
                    self.logger.info(f"[PARSE DEBUG] Tool: {tc.get('function', {}).get('name')}")
            else:
                self.logger.info(f"[PARSE DEBUG] No tool calls found in output")
        
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
        # DEBUG: Log tools being passed
        self.logger.info(f"[TEMPLATE DEBUG] _apply_chat_template called with {len(tools) if tools else 0} tools")
        if tools:
            tool_names = [t.get('function', {}).get('name', 'unknown') for t in tools]
            self.logger.info(f"[TEMPLATE DEBUG] Tool names: {tool_names}")
            self.logger.info(f"[TEMPLATE DEBUG] First tool schema: {json.dumps(tools[0], indent=2)[:500]}")
        
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
                        self.logger.info(f"[TEMPLATE DEBUG] Attempting to pass tools to apply_chat_template")
                        template_kwargs['tools'] = tools
                        prompt = self.tokenizer.apply_chat_template(
                            formatted_messages,
                            tokenize=False,
                            **template_kwargs
                        )
                        self.logger.info(f"[TEMPLATE DEBUG] Successfully applied template with tools")
                        self.logger.info(f"[TEMPLATE DEBUG] Prompt preview (first 500 chars):\n{prompt[:500]}")
                        return prompt
                    except Exception as e:
                        self.logger.warning(f"[TEMPLATE DEBUG] Failed to pass tools to template: {e}")
                        # If tools not supported in template, add to system message
                        tools_description = self._format_tools_as_text(tools)
                        if formatted_messages and formatted_messages[0]['role'] == 'system':
                            formatted_messages[0]['content'] += f"\n\n{tools_description}"
                        else:
                            formatted_messages.insert(0, {
                                'role': 'system',
                                'content': tools_description
                            })
                        self.logger.info(f"[TEMPLATE DEBUG] Added tools as system message fallback")
                
                prompt = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    **template_kwargs
                )
                self.logger.info(f"[TEMPLATE DEBUG] Prompt preview (first 500 chars):\n{prompt[:500]}")
                return prompt
            except Exception as e:
                self.logger.warning(f"Chat template failed: {e}, falling back to manual formatting")
        
        # Fallback: Manual formatting
        self.logger.info(f"[TEMPLATE DEBUG] Using manual chat formatting fallback")
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
    
    def _truncate_messages_if_needed(self, messages: List[Dict[str, str]], 
                                     tools: Optional[List[Dict]] = None) -> List[Dict[str, str]]:
        """Truncate messages to prevent context overflow.
        
        Strategy:
        1. Keep system message (if exists)
        2. Keep last user message
        3. Remove older messages if still too long
        
        Note: Individual tool results are NOT truncated here.
        Final prompt-level truncation in _ensure_prompt_within_limit handles overflow.
        
        Args:
            messages: List of chat messages
            tools: Available tools (for rough size estimation)
            
        Returns:
            Truncated messages list
        """
        if not messages:
            return messages
        
        # Make a copy to avoid modifying original
        messages = [msg.copy() for msg in messages]
        
        # Estimate total tokens (rough approximation)
        # We'll do a more precise check after template application
        total_chars = sum(len(msg.get('content', '')) for msg in messages)
        
        # If roughly within limits, return (4 chars ≈ 1 token rough estimate)
        if total_chars // 4 < self.max_context_tokens * 0.8:
            return messages
        
        # Need to remove some messages
        # Keep: system (if exists), last user message
        # Remove: older messages, starting from oldest
        
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
            # Remove messages before last user message, keeping some context
            # Keep at most 5 messages before last user message
            keep_from = max(0, last_user_idx - 5)
            messages = messages[keep_from:]
            self.logger.warning(f"[CONTEXT] Removed {keep_from} older messages to fit context")
        
        # Re-add system message at the beginning
        if system_msg:
            messages.insert(0, system_msg)
        
        return messages
    
    def _ensure_prompt_within_limit(self, prompt: str) -> str:
        """Ensure final prompt is within token limit.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Truncated prompt if necessary
        """
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        if len(tokens) <= self.max_context_tokens:
            # Within limit, no action needed
            return prompt
        
        # Truncate from the middle, keeping start (system/tools) and end (recent context)
        self.logger.warning(
            f"[CONTEXT] Prompt exceeds limit: {len(tokens)} > {self.max_context_tokens}, truncating..."
        )
        
        # Keep first 30% and last 60% of tokens
        keep_start = int(self.max_context_tokens * 0.3)
        keep_end = int(self.max_context_tokens * 0.6)
        
        truncated_tokens = tokens[:keep_start] + tokens[-(keep_end - 1):]  # -1 for '...' token
        
        # Insert truncation marker
        ellipsis_token = self.tokenizer.encode("...", add_special_tokens=False)
        truncated_tokens = tokens[:keep_start] + ellipsis_token + tokens[-keep_end:]
        
        # Ensure we're still within limit
        if len(truncated_tokens) > self.max_context_tokens:
            truncated_tokens = truncated_tokens[:self.max_context_tokens]
        
        truncated_prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        self.logger.warning(
            f"[CONTEXT] Truncated to {len(truncated_tokens)} tokens "
            f"(removed ~{len(tokens) - len(truncated_tokens)} tokens)"
        )
        
        return truncated_prompt
    
    def _parse_tool_calls(self, text: str, tools: List[Dict]) -> tuple[Optional[List[Dict]], str]:
        """Parse tool calls from model output.
        
        Supports multiple formats:
        1. XML-wrapped: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        2. Function tag format: <function=tool_name>{...arguments...}</function>
        3. Plain JSON with tool_name: {"tool_name": "...", "arguments": {...}}
        4. Plain JSON with name: {"name": "...", "arguments": {...}}
        5. Array of tool calls
        
        Args:
            text: Generated text from model
            tools: Available tools
            
        Returns:
            Tuple of (tool_calls, remaining_text)
        """
        tool_calls = []
        remaining_text = text
        
        # Pattern 1: XML-wrapped tool calls (most common for Qwen)
        # Matches: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        xml_pattern = r'<tool_call>\s*(\{[^<]+\})\s*</tool_call>'
        
        # Pattern 2: Function tag format
        # Matches: <function=tool_name>{...arguments...}</function>
        function_tag_pattern = r'<function=([^>]+)>(\{[^<]+\})</function>'
        
        # Pattern 3: Single tool call with tool_name
        single_tool_name_pattern = r'\{"tool_name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]+\})\}'
        
        # Pattern 4: Single tool call with name
        single_name_pattern = r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]+\})\}'
        
        # Pattern 5: Array of tool calls
        array_pattern = r'\[\s*\{[^\]]+\}\s*\]'
        
        try:
            # Try function tag format first (for models like Midm)
            # Matches: <function=tool_name>{...arguments...}</function>
            function_matches = re.finditer(function_tag_pattern, text, re.DOTALL)
            function_found = False
            
            for match in function_matches:
                function_found = True
                tool_name = match.group(1).strip()
                arguments_str = match.group(2).strip()
                
                try:
                    # Parse arguments JSON
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
                    self.logger.warning(f"Failed to parse function tag arguments JSON: {e}")
                    continue
            
            if function_found:
                # Remove all function tags from text
                remaining_text = re.sub(function_tag_pattern, '', text, flags=re.DOTALL).strip()
                return (tool_calls if tool_calls else None), remaining_text
            
            # Try XML pattern (most common for Qwen)
            xml_matches = re.finditer(xml_pattern, text, re.DOTALL)
            xml_found = False
            
            for match in xml_matches:
                xml_found = True
                json_str = match.group(1)
                try:
                    call_data = json.loads(json_str)
                    # Support both 'name' and 'tool_name' keys
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
                    self.logger.warning(f"Failed to parse XML tool call JSON: {e}")
                    continue
            
            if xml_found:
                # Remove all XML tool call tags from text
                remaining_text = re.sub(xml_pattern, '', text, flags=re.DOTALL).strip()
                return (tool_calls if tool_calls else None), remaining_text
            
            # Try array pattern
            array_match = re.search(array_pattern, text, re.DOTALL)
            if array_match:
                json_str = array_match.group(0)
                calls_data = json.loads(json_str)
                
                for call_data in calls_data:
                    # Support both 'name' and 'tool_name' keys
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
                
                # Remove the JSON array from text
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
                
                # Remove the JSON from text
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
                
                # Remove the JSON from text
                remaining_text = text[:single_match.start()] + text[single_match.end():]
                return (tool_calls if tool_calls else None), remaining_text.strip()
        
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse tool calls: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing tool calls: {e}")
        
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
