# Glossary — Every Technical Term in Plain English

## Core concepts

**Code hallucination:** When an AI writes code that looks correct but is actually wrong — inventing functions that don't exist, importing nonexistent libraries, or using wrong parameter names. The code compiles and looks plausible, but it fails when you actually run it.

**Repository (repo):** A folder containing all the files for a software project. Could be 5 files or 5,000 files. "Repository-level code generation" means the AI needs to write code that works with all the existing files in the project.

**Context:** The code snippets you include in the AI's prompt to help it write correct code. Like giving someone reference materials before asking them to write something.

**Context selection:** Choosing which pieces of the repository to show the AI. You can't show it everything (prompt size limits), so you have to pick the most useful pieces.

**RAG (Retrieval-Augmented Generation):** A technique where you first search for relevant documents, then include them in the AI's prompt before asking it to generate. "Retrieval" = find relevant stuff. "Augmented" = add it to the prompt. "Generation" = the AI writes its response.

## The four hallucination types

**Resource hallucination:** The AI imports something that doesn't exist. `import torch.quantum` — this module was never written. Python throws `ImportError` or `ModuleNotFoundError`.

**Naming hallucination:** The AI uses a variable or function name that doesn't exist in the codebase. `result = get_data()` when the actual function is called `fetch_data()`. Python throws `NameError`.

**Mapping hallucination:** The AI calls a real function but with the wrong arguments or in the wrong way. `torch.cat(x, y)` when the correct call is `torch.cat([x, y], dim=0)`. Python throws `TypeError` or `AttributeError`.

**Logic hallucination:** The code runs without crashing but produces the wrong result. An off-by-one error, a wrong condition, or a backwards comparison. Python throws `AssertionError` when the test cases check the output.

## ML / training concepts

**Embedding:** Converting text (or code) into a fixed-size list of numbers that captures its meaning. CodeBERT converts any code snippet into 768 numbers. Similar code gets similar numbers, so you can compare code by comparing their number lists.

**Vector:** Just a list of numbers. A 768-dimensional vector is a list of 768 numbers. "Embedding" and "vector" are used interchangeably in this context.

**Cosine similarity:** A way to measure how similar two vectors are. It computes the angle between them — vectors pointing in the same direction have high similarity (close to 1.0), vectors pointing in opposite directions have low similarity (close to -1.0). Used by traditional RAG to find "similar" code.

**Contrastive learning:** A training method where you show the model pairs of examples — a good one and a bad one — and train it to tell them apart. "Here's context that led to correct code (good), and here's context that led to a hallucination (bad). Learn to score the good one higher."

**Triplet:** A training example with three parts: (query, positive_context, negative_context). The query is the coding task. The positive context led to correct code. The negative context led to a hallucination.

**InfoNCE loss:** The specific math formula used to train contrastive models. It says: "the score of the positive example should be much higher than the score of any negative example." If the model gets confused and scores a negative higher than the positive, the loss is large and the model gets corrected.

**Temperature (tau, τ):** A number (typically 0.07) that controls how "sharp" the scoring is. Low temperature = model is very confident, small differences in score become large differences in probability. High temperature = model is more uncertain, scores are more spread out.

**MLP (Multi-Layer Perceptron):** The simplest type of neural network. Input numbers → multiply by weights → add bias → activation function → multiply by more weights → output. Just stacked layers of basic math operations.

**Frozen encoder:** Using a pre-trained model (like CodeBERT) without changing any of its weights. You run it to get embeddings, but you don't train it further. "Frozen" = its parameters are locked, not updated during training.

**Fine-tuning:** Taking a pre-trained model and continuing to train it on your specific task. This updates the model's weights. More powerful than freezing, but needs more compute.

**Epoch:** One complete pass through all training data. If you have 3,000 triplets and train for 10 epochs, the model sees each triplet 10 times.

**Batch size:** How many training examples you process at once before updating the model's weights. Batch size 32 = look at 32 triplets, compute the average loss, update weights once.

**Learning rate:** How big a step the model takes when updating its weights. Too large = unstable, overshoots. Too small = trains too slowly. Typical: 2e-5 (0.00002).

## Model names

**CodeBERT:** A 125 million parameter model trained by Microsoft to understand code. It reads code and produces embeddings. It does NOT generate code — it only understands it. Used in this project as a feature extractor for the HCCS scorer.

**DeepSeek-Coder:** A family of code generation models. The 1.3B version fits easily on a T4 GPU. The 6.7B version fits with 4-bit quantization. These are the models that actually write the code.

**StarCoder2:** Another code generation model from BigCode. The paper uses it as one of three LLMs for generating training data. We may use it as an alternative to DeepSeek-Coder.

## Infrastructure terms

**GPU (Graphics Processing Unit):** A chip designed for parallel math operations. Originally for graphics, now essential for AI because neural networks are just massive parallel math. A T4 GPU has 16GB of dedicated memory.

**VRAM:** The GPU's own memory. Separate from your laptop's RAM. Models must fit in VRAM to run on the GPU. CodeBERT (125M params) needs ~500MB. DeepSeek-Coder-1.3B needs ~3GB. DeepSeek-Coder-6.7B needs ~14GB (or ~4GB with 4-bit quantization).

**T4 GPU:** The GPU you get with Colab Pro (free for students). 16GB VRAM, good enough for all our tasks. Fine-tuning CodeBERT: fits. Running DeepSeek-1.3B: fits. Running DeepSeek-6.7B with quantization: fits.

**A100 GPU:** A much more powerful GPU available on Colab Pro. 40GB or 80GB VRAM. We probably don't need it, but it would speed up data generation.

**Quantization (4-bit):** Compressing a model's weights from 32-bit or 16-bit numbers down to 4-bit numbers. Makes the model ~4x smaller in memory, with a small quality loss. Allows running 6.7B models on a 16GB T4.

**Compute units (CU):** Colab Pro's billing currency. T4 costs ~2 CU/hour. A100 costs ~15 CU/hour. You get 100 CU/month. Our project needs ~50-70 CU total.

**Sandbox / sandboxed execution:** Running untrusted code in an isolated environment so it can't damage your system. On Colab, we use subprocess with timeouts. The code runs, we capture its output, and if it goes rogue (infinite loop, memory hog), we kill it after 30 seconds.

## Evaluation terms

**Baseline:** A simple method you compare your approach against. If your fancy system can't beat BM25 (a 1990s keyword search algorithm), something is wrong.

**Ablation study:** Testing your system with individual components removed to see how much each one contributes. "Full system: 85% pass rate. Without EFL: 78%. Without router: 80%. Without HCCS: 65%." This shows the EFL contributes 7%, the router contributes 5%, and the HCCS contributes 13%.

**Pass@k:** "If the AI gets k attempts, does at least one work?" Pass@1 = the first attempt must work. Pass@5 = at least one of five attempts must work. We focus on Pass@1.

**Hallucination Rate (HR):** Percentage of generated code samples that contain at least one hallucination. Lower is better. HR = hallucinating_samples / total_samples.

**Hallucination Reduction Ratio:** How much better you are than the baseline. (HR_baseline - HR_ours) / HR_baseline. If baseline has 40% HR and you have 20%, your reduction ratio is 50%.

## Benchmark datasets

**CodeHaluEval:** Our primary benchmark. 8,883 code samples across 699 tasks. Each sample has: a coding query, repository context, test cases, and hallucination type labels. Created by Tian et al. (2025). Available on HuggingFace.

**HumanEval:** 164 hand-written Python programming problems by OpenAI. The most widely-used code generation benchmark. We use it to show our method doesn't hurt general code quality.

**MBPP (Mostly Basic Programming Problems):** 974 crowd-sourced Python tasks. Simpler than HumanEval but larger. Used as a secondary quality check.

**CrossCodeEval:** A multilingual benchmark for cross-file code completion. Stretch goal only — probably won't get to it in 2 weeks.

## Search algorithms

**BM25:** A classic text search algorithm from the 1990s. Ranks documents by keyword overlap with the query, weighted by word rarity. "fetch" appearing in a chunk about fetching weather data scores high. Fast, simple, no AI needed. Our baseline to beat.

**Cosine similarity retrieval:** Embed the query and all chunks using CodeBERT, then rank chunks by cosine similarity to the query. This is what "traditional RAG" does. Better than BM25 (understands meaning, not just keywords) but still misses the hallucination-prevention signal.
