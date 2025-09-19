import copy
import re

import torch
import torch.nn as nn
import yaml
from bert_score import score
from models.model_base import PreTrainedModelWrapper
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, RagTokenForGeneration, RagRetriever


class AutoDPOModelForCausalLM(PreTrainedModelWrapper):
    """
    An autoregressive model with support for custom modules in addition to the language model.
    This class inherits from `PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained`, and `generate`. To call a method of the wrapped
    model, simply manipulate the `pretrained_model` attribute of this class.

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the custom module class you designed. Currently, the supported args are: ______
    """

    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]

    ####################################################################################
    # TODO (Optional): Please put any required arguments for your custom module here
    supported_args = ()
    ####################################################################################

    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to any `CustomModule` class.
        """
        pretrained_model.eval()
        super().__init__(pretrained_model, **kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure CustomModule is repalced with the name of your custom module class
        # Remember that the below lines are just an example
        # You can reanme the class and the variabels to fit your custom module name,
        # just make sure they are consistent in the code
        # =========================================================================================
        # custom_module_kwargs, _, _ = self._split_kwargs(kwargs)
        # self.custom_module = CustomModule(self.pretrained_model.config, **custom_module_kwargs)
        # self._init_weights(**custom_module_kwargs)
        self.current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model = self.pretrained_model.to(self.current_device)

        main_config = {}
        with open("./main_config.yaml") as f:
            try:
                main_config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading main_config.yaml: {e}! Please check the file format.")

        print("model name:", pretrained_model.config._name_or_path)
        self.is_rag = ("rag" in main_config.get("eval_method", [])
                       and "rag" in pretrained_model.config._name_or_path)

        print(f"Is RAG: {self.is_rag}")

        if self.is_rag:
            self.rag_question_encoder = RagTokenForGeneration.from_pretrained("rag-token-nq",
                                                                              use_dummy_dataset=True,
                                                                              resume_download=None).question_encoder
            self.rag_tokenizer = AutoTokenizer.from_pretrained("rag-token-nq", resume_download=None)
            self.retriever = RagRetriever.from_pretrained(
                "rag-token-nq", index_name="exact", use_dummy_dataset=True, resume_download=None
            )

        ###########################################################################################

    def _init_weights(self, **kwargs):
        """
        Initializes the weights of the custom module. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `custom_module_init_strategy`
        argument when calling `.from_pretrained`. Supported strategies are:
            - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `CustomModule` class.
        """
        ###############################################################
        # TODO (Optional): Please implement the initialization strategy for your custom module here
        pass
        ###############################################################

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the custom module
        to the state dictionary of the wrapped model by prepending the key with `custom_module.`.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state_dict = {}

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure "custom_module" is repalced with the name of your custom module class
        # =========================================================================================
        # custom_module_state_dict = self.custom_module.state_dict(*args, **kwargs)
        # for k, v in custom_module_state_dict.items():
        #     pretrained_model_state_dict[f"custom_module.{k}"] = v
        ###########################################################################################
        return pretrained_model_state_dict

    def post_init(self, state_dict):
        """
        We add the state dictionary of the custom module to the state dictionary of the wrapped model
        by prepending the key with `custom_module.`. This function removes the `custom_module.` prefix from the
        keys of the custom module state dictionary.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not hasattr(self, 'custom_module'):
            return

        for k in list(state_dict.keys()):
            if "custom_module." in k:
                state_dict[k.replace("custom_module.", "")] = state_dict.pop(k)
        self.custom_module.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for CustomModule models."
                )

            # get the lm_head device
            for name, module in self.pretrained_model.named_modules():
                if any(attribute in name for attribute in self.lm_head_namings):
                    lm_head_device = module.weight.device
                    break

            # put custom_module on the same device as the lm_head to avoid issues
            self.custom_module = self.custom_module.to(lm_head_device)

            def set_device_hook(module, input, outputs):
                r"""
                A hook that sets the device of the output of the model to the device of the first
                parameter of the model.

                Args:
                    module (`nn.Module`):
                        The module to which the hook is attached.
                    input (`tuple`):
                        The input to the module.
                    outputs (`tuple`):
                        The output of the module.
                """
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True

    def push_to_hub(self, *args, **kwargs):
        """Push the model to the Hugging Face hub."""
        ###########################################################################################
        # TODO (Optional): Please uncomment the following line to add the custom module to the hub model
        # Make sure custom_module is repalced with the name of your custom module class
        # =========================================================================================
        # self.pretrained_model.custom_module = self.custom_module
        ###########################################################################################

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        """
        Applies a forward pass to the wrapped model and returns the output from the model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        Returns:
            output_dict (`dict`): A dictionary containing the output from the model.
        """
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        output_dict = {}

        ###############################################################
        # TODO: Please implement your customized forward pass here
        # =============================================================
        # put model on device
        input_ids = input_ids.to(self.current_device)
        attention_mask = attention_mask.to(self.current_device)

        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        output_dict["loss"] = outputs.loss
        output_dict["logits"] = outputs.logits
        output_dict["past_key_values"] = outputs.past_key_values
        output_dict["hidden_states"] = outputs.hidden_states
        output_dict["attentions"] = outputs.attentions
        output_dict["cross_attentions"] = outputs.cross_attentions
        ###############################################################

        return output_dict

    def get_logprobs(self, batch, tokenizer):
        """
        Computes the log probabilities of a response using the model respectively.

        Args:
            batch (`dict` of `list`): A dictionary containing the input data for the DPO model.
                The data format is as follows:
                {
                    "prompt": List[str],
                    "chosen": List[str],
                    "rejected": List[str],
                    "chosen_logps": Optional(torch.FloatTensor)
                    "rejected_logps": Optional(torch.FloatTensor)
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input data.
        Returns:
            A tuple of two tensors: (chosen_logps, rejected_logps)
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the chosen responses. Shape: (batch_size,)
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the rejected responses. Shape: (batch_size,)
        """
        ###############################################################
        # TODO: Please implement your customized logprob computation here
        # =============================================================
        tokenizer.pad_token = tokenizer.eos_token

        combined_chosen = [batch["prompt"][i] + " " + batch["chosen"][i] for i in range(len(batch["chosen"]))]
        combined_rejected = [batch["prompt"][i] + " " + batch["rejected"][i] for i in range(len(batch["rejected"]))]

        max_len_in_batch = max([len(combined_chosen[i]) for i in range(len(combined_chosen))])
        max_len_in_batch_rej = max([len(combined_rejected[i]) for i in range(len(combined_rejected))])
        max_len = min(1024, max(max_len_in_batch, max_len_in_batch_rej))
        chosen_tokens = tokenizer(combined_chosen, return_tensors="pt", padding="max_length", truncation=True,
                                  max_length=max_len)
        rejected_tokens = tokenizer(combined_rejected, return_tensors="pt", padding="max_length", truncation=True,
                                    max_length=max_len)

        chosen_pad_mask = torch.tensor(chosen_tokens["attention_mask"], device=self.current_device)
        rejected_pad_mask = torch.tensor(rejected_tokens["attention_mask"], device=self.current_device)

        # chosen tokenized and lengths
        prompt_only_token_ids = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True).input_ids
        prompt_only_len = prompt_only_token_ids.shape[1] + 1  # +1 for the " " token

        # forward pass
        with torch.no_grad():
            chosen_logits = \
            self.forward(input_ids=chosen_tokens["input_ids"], attention_mask=chosen_tokens["attention_mask"])["logits"]
            rejected_logits = \
            self.forward(input_ids=rejected_tokens["input_ids"], attention_mask=rejected_tokens["attention_mask"])[
                "logits"]

        # extract actual chosen and rejected tokens
        extracted_chosen = chosen_logits[:, prompt_only_len:, :]  # get only response from logits
        extracted_rejected = rejected_logits[:, prompt_only_len:, :]  # get only response from logits

        # set pad logits to 0
        chosen_pad_mask_expanded = chosen_pad_mask.unsqueeze(-1)[:, prompt_only_len:, :]
        rejected_pad_mask_expanded = rejected_pad_mask.unsqueeze(-1)[:, prompt_only_len:, :]

        # Apply the mask to the logits
        filtered_chosen_logits = extracted_chosen * chosen_pad_mask_expanded
        filtered_rejected_logits = extracted_rejected * rejected_pad_mask_expanded
        # At this point, all the logits corresponding to a pad elem are set to zero

        # now, per batch, do the following:
        #   for each element of chosen_only, look for the corresponding element in filtered_chosen_logits of that batch
        #   use the logits of all those tokens and put it in a tensor
        #   so we have one tensor of size (seq_len)
        #   then we sum them together
        #   then we divide the sum by the number of non-zero elements
        # so, we end up with a tensor of size (batch_size)
        chosen_logps = []
        rejected_logps = []
        for batch_idx in range(filtered_chosen_logits.size(0)):
            # for chosen logprobs
            filtered_chosen_log_probs = nn.functional.log_softmax(filtered_chosen_logits[batch_idx], dim=-1)

            selected_filtered_chosen_log_probs = [
                filtered_chosen_log_probs[i][chosen_tokens.input_ids[batch_idx][prompt_only_len + i]]
                for i in range(len(filtered_chosen_logits[batch_idx]))]
            chosen_logps.append(sum(selected_filtered_chosen_log_probs) / torch.sum(chosen_pad_mask[batch_idx]))

            # for rejected logprobs
            filtered_rejected_log_probs = nn.functional.log_softmax(filtered_rejected_logits[batch_idx], dim=-1)
            selected_filtered_rejected_log_probs = [
                filtered_rejected_log_probs[i][rejected_tokens.input_ids[batch_idx][prompt_only_len + i]]
                for i in range(len(filtered_rejected_logits[batch_idx]))]
            rejected_logps.append(sum(selected_filtered_rejected_log_probs) / torch.sum(rejected_pad_mask[batch_idx]))

        ###############################################################

        return torch.tensor(chosen_logps), torch.tensor(rejected_logps)

    def prediction_step_reward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
        """
        Computes the reward socres of the chosen and reject responses by implementing the DPO reward function
        Reference of the DPO reward function: https://arxiv.org/pdf/2305.18290.pdf

        Args:
            policy_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        Returns:
            output_dict (`dict`):
                A dictionary containing the reward scores of the chosen and rejected responses.
        """
        output_dict = {
            "chosen_rewards": [],
            "rejected_rewards": []
        }

        ########################################################################
        # TODO: Please implement the prediction step that computes the rewards
        # ======================================================================
        # You need to return one reward score for each chosen and rejected response.
        # ======================================================================
        beta = 0.1  # same beta as used in training (dpo_script.py)

        chosen_rewards = -beta * (policy_chosen_logps - reference_chosen_logps).detach()
        output_dict["chosen_rewards"] = chosen_rewards
        output_dict["chosen_rewards"] = output_dict["chosen_rewards"].cpu()

        rejected_rewards = -beta * (policy_rejected_logps - reference_rejected_logps).detach()
        output_dict["rejected_rewards"] = rejected_rewards.cpu()
        ########################################################################

        return output_dict

    def prediction_step_mcqa(self, batch, tokenizer):
        """
        Computes the mcqa prediction of the given question.

        Args:
            batch (`dict` of `list`):
                A dictionary containing the input mcqa data for the DPO model.
                The data format is as follows:
                {
                    "question": List[str], each <str> contains the question body and the choices
                    "answer": List[str], each <str> is a single letter representing the correct answer
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input questions.
        Returns:
            output_dict (`dict`): A dictionary containing the model predictions given input questions.
        """
        output_dict = {"preds": []}

        ########################################################################
        # TODO: Please implement the prediction step that generates the prediction of the given MCQA question
        # ======================================================================
        # You need to return one letter prediction for each question.
        # ======================================================================
        original_batch = copy.deepcopy(batch)
        if self.is_rag:
            for i in range(len(batch["question"])):
                question = batch["question"][i]
                input_ids = self.rag_tokenizer(question, return_tensors="pt").input_ids
                question_hidden_states = self.rag_question_encoder(input_ids)[0]

                docs_dict = self.retriever(input_ids.numpy(), question_hidden_states.detach().numpy(),
                                           return_tensors="pt")
                retrieved_docs = self.rag_tokenizer.batch_decode(docs_dict["context_input_ids"],
                                                                 skip_special_tokens=True)

                input_text = retrieved_docs[0]  # play around with nb of docs, sizes of the chunks, ...

                batch["question"][i] = input_text

        tokenizer.pad_token = tokenizer.eos_token
        max_new_tokens = 50

        max_len_in_batch = max([len(batch["question"][i]) for i in range(len(batch["question"]))])
        max_len = min(1024, max_len_in_batch)

        # TODO: check if we need to generate separately for each batch
        tokens = tokenizer(batch["question"], return_tensors="pt", padding="max_length", truncation=True,
                           max_length=max_len - max_new_tokens)

        input_ids = tokens["input_ids"].to(self.current_device)
        attention_mask = tokens["attention_mask"].to(self.current_device)

        with torch.no_grad():
            generated_ids = self.pretrained_model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                           max_new_tokens=max_new_tokens)

        generated_answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i, answer in enumerate(generated_answers):
            original_question = original_batch["question"][i]
            question = batch["question"][i]
            answer_choice = self._extract_mcqa_choice_from_answer(question=original_question,
                                                                  generated_answer=answer[len(question):].lstrip(),
                                                                  tokenizer=tokenizer)

            output_dict["preds"].append(answer_choice)
        ########################################################################

        return output_dict


    def _extract_mcqa_choice_from_answer(self, question, generated_answer, tokenizer):
        """
        Extracts the choice from the answer string.

        Args:
            question (`str`): The question string.
            generated_answer (`str`): The generated answer string.
        Returns:
            choice (`str`): The choice extracted from the answer string.
        """
        answer_options, option_letters, question_part = self._extract_options(question)

        bert_scores = {}
        for i, option in enumerate(answer_options):
            # TODO: check what makes most sense for the score computation (e.g., using the question or not, ...)
            precision, recall, f1 = score(
                [generated_answer],
                [option],
                lang="en",
                model_type="bert-base-uncased",
                rescale_with_baseline=True)
            bert_scores[option_letters[i]] = f1[0].item()

        best_letter = max(bert_scores, key=bert_scores.get)

        return best_letter


    def _extract_options(self, question):
        """
        Extracts the answer options and corresponding answers (i.e., the lettres) from the input string.

        Args:
            question (`str`): The input string containing the answer options.
        Returns:
            options (`list` of `str`): The list of options.
            option_letters (`list` of `str`): The list of letters corresponding to the options.
            question_part (`str`): The question part of the input string.
        """

        # extract the question and answer parts separately
        pattern = r'\n\nOptions:\n(.+?)\n\nAnswer:'
        match = re.search(pattern, question, re.DOTALL)

        if match:
            question_part = match.group(0)
            options_part = match.group(1)

            options = options_part.strip().split('\n')
            option_letters = [option[0] for option in options]

            return options, option_letters, question_part
        else:
            # answer_options, option_letters, question_part
            return ["A. ", "B. ", "C. ", "D. "], ["A", "B", "C", "D"], ""

class AutoDPOModelForSeq2SeqLM(PreTrainedModelWrapper):
    r"""
    A seq2seq model with support for custom modules in addition to the transformer model.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained` and `push_to_hub` and also provides some additional
    functionalities such as `generate`.

    Args:
        pretrained_model (`transformers.PreTrainedModel`):
            The model to wrap. It should be a causal language model such as GPT2.
            or any model mapped inside the `AutoModelForSeq2SeqLM` class.
        kwargs:
            Additional keyword arguments passed along to any `CustomModule` classes.
    """

    transformers_parent_class = AutoModelForSeq2SeqLM
    lm_head_namings = ["lm_head", "embed_out", "output_projection"]
    ####################################################################################
    # TODO (Optional): Please put any required arguments for your custom module here
    supported_args = ()
    ####################################################################################

    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        self.is_encoder_decoder = True
        if not self._has_lm_head():
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure CustomModule is repalced with the name of your custom module class
        # Remember that the below lines are just an example
        # You can reanme the class and the variabels to fit your custom module name,
        # just make sure they are consistent in the code
        # =========================================================================================
        # custom_module_kwargs, _, _ = self._split_kwargs(kwargs)
        # self.custom_module = CustomModule(self.pretrained_model.config, **custom_module_kwargs)
        # self._init_weights(**custom_module_kwargs)
        ###########################################################################################

    def _has_lm_head(self):
        # check module names of all modules inside `pretrained_model` to find the language model head
        for name, _module in self.pretrained_model.named_modules():
            if any(attribute in name for attribute in self.lm_head_namings):
                return True
        return False

    def _init_weights(self, **kwargs):
        """
        Initializes the weights of the custom module. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `custom_module_init_strategy`
        argument when calling `.from_pretrained`. Supported strategies are:
            - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `CustomModule` class.
        """
        ###############################################################
        # TODO (Optional): Please implement the initialization strategy for your custom module here
        pass
        ###############################################################

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the custom module
        to the state dictionary of the wrapped model by prepending the key with `custom_module.`.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state_dict = {}

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure "custom_module" is repalced with the name of your custom module class
        # =========================================================================================
        # custom_module_state_dict = self.custom_module.state_dict(*args, **kwargs)
        # for k, v in custom_module_state_dict.items():
        #     pretrained_model_state_dict[f"custom_module.{k}"] = v
        ###########################################################################################
        return pretrained_model_state_dict

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the custom module to the state dictionary of the wrapped model
        by prepending the key with `custom_module.`. This function removes the `custom_module.` prefix from the
        keys of the custom module state dictionary.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not hasattr(self, 'custom_module'):
            return

        for k in list(state_dict.keys()):
            if "custom_module." in k:
                state_dict[k.replace("custom_module.", "")] = state_dict.pop(k)
        self.custom_module.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for CustomModule models."
                )

            # get the lm_head device
            for name, module in self.pretrained_model.named_modules():
                if any(attribute in name for attribute in self.lm_head_namings):
                    lm_head_device = module.weight.device
                    break

            # put custom_module on the same device as the lm_head to avoid issues
            self.custom_module = self.custom_module.to(lm_head_device)

            def set_device_hook(module, input, outputs):
                r"""
                A hook that sets the device of the output of the model to the device of the first
                parameter of the model.

                Args:
                    module (`nn.Module`):
                        The module to which the hook is attached.
                    input (`tuple`):
                        The input to the module.
                    outputs (`tuple`):
                        The output of the module.
                """
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True

    def push_to_hub(self, *args, **kwargs):
        """Push the model to the Hugging Face hub."""
        ###########################################################################################
        # TODO (Optional): Please uncomment the following line to add the custom module to the hub model
        # Make sure custom_module is repalced with the name of your custom module class
        # =========================================================================================
        # self.pretrained_model.custom_module = self.custom_module
        ###########################################################################################

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the output from the model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        Returns:
            ouput_dict (`dict`): A dictionary containing the output from the model.
        """
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        ouput_dict = {}

        ###############################################################
        # TODO: Please implement your customized forward pass here
        # =============================================================
        raise NotImplementedError
        ###############################################################

        return ouput_dict

    def get_logprobs(self, batch, tokenizer):
        """
        Computes the log probabilities of a response using the model respectively.

        Args:
            batch (`dict` of `list`): A dictionary containing the input data for the DPO model.
                The data format is as follows:
                {
                    "prompt": List[str],
                    "chosen": List[str],
                    "rejected": List[str],
                    "chosen_logps": Optional(torch.FloatTensor)
                    "rejected_logps": Optional(torch.FloatTensor)
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input data.
        Returns:
            A tuple of two tensors: (chosen_logps, rejected_logps)
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the chosen responses. Shape: (batch_size,)
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the rejected responses. Shape: (batch_size,)
        """
        ###############################################################
        # TODO: Please implement your customized logprob computation here
        # =============================================================
        raise NotImplementedError
        ###############################################################

        return chosen_logps, rejected_logps

    def prediction_step_reward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
        """
        Computes the reward socres of the chosen and reject responses by implementing the DPO reward function
        Reference of the DPO reward function: https://arxiv.org/pdf/2305.18290.pdf

        Args:
            policy_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        Returns:
            output_dict (`dict`):
                A dictionary containing the reward scores of the chosen and rejected responses.
        """
        output_dict = {
            "chosen_rewards": [],
            "rejected_rewards": []
        }

        ########################################################################
        # TODO: Please implement the dpo loss function to compute the rewards
        # You need to return one reward score for each chosen and rejected response.
        # ======================================================================
        raise NotImplementedError
        ########################################################################

        return output_dict

    def prediction_step_mcqa(self, batch, tokenizer):
        """
        Computes the mcqa prediction of the given question.

        Args:
            batch (`dict` of `list`):
                A dictionary containing the input mcqa data for the DPO model.
                The data format is as follows:
                {
                    "question": List[str], each <str> contains the question body and the choices
                    "answer": List[str], each <str> is a single letter representing the correct answer
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input questions.
        Returns:
            output_dict (`dict`): A dictionary containing the model predictions given input questions.
        """
        output_dict = {"preds": []}

        ########################################################################
        # TODO: Please implement the prediction step that generates the prediction of the given MCQA question
        # ======================================================================
        # You need to return one letter prediction for each question.
        # ======================================================================
        raise NotImplementedError
        ########################################################################

        return output_dict
