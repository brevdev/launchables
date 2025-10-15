{
  "cells": [
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "xFCx6jZU3m11"
      },
      "source": [
        "<!-- Banner Image -->\n",
        "<img src=\"https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/brevdevnotebooks.png\" width=\"100%\">\n",
        "\n",
        "<!-- Links -->\n",
        "<center>\n",
        "  <a href=\"https://brev.nvidia.com\" style=\"color: #06b6d4;\">Console</a> •\n",
        "  <a href=\"https://developer.nvidia.com/brev\" style=\"color: #06b6d4;\">Docs</a> •\n",
        "  <a href=\"/\" style=\"color: #06b6d4;\">Templates</a> •\n",
        "  <a href=\"https://discord.gg/NVDyv7TUgJ\" style=\"color: #06b6d4;\">Discord</a>\n",
        "</center>\n",
        "\n",
        "# Fine-tuning Mistral on your own data 🤙\n",
        "\n",
        "Welcome!\n",
        "\n",
        "In this notebook and tutorial, we will fine-tune the [Mistral 7B](https://github.com/mistralai/mistral-src) model - which outperforms Llama 2 13B on all tested benchmarks - ***on your own data!***\n",
        "\n",
        "## Watch the accompanying video walk-through [here](https://youtu.be/kmkcNVvEz-k?si=Ogt1wRFNqYI6zXfw&t=1)!\n",
        "\n",
        "I did this for **just one dollar ($1)** on an 1x A10G 24GB from Brev (instructions below).\n",
        "\n",
        "This tutorial will use QLoRA, a fine-tuning method that combines quantization and LoRA. For more information about what those are and how they work, see [this post](https://developer.nvidia.com/brev/blog/how-qlora-works).\n",
        "\n",
        "In this notebook, we will load the large model in 4bit using `bitsandbytes` and use LoRA to train using the PEFT library from Hugging Face 🤗.\n",
        "\n",
        "Note that if you ever have trouble importing something from Huggingface, you may need to run `huggingface-cli login` in a shell. To open a shell in Jupyter Lab, click on 'Launcher' (or the '+' if it's not there) next to the notebook tab at the top of the screen. Under \"Other\", click \"Terminal\" and then run the command.\n",
        "\n",
        "### Help us make this tutorial better! Please provide feedback on the [Discord channel](https://discord.gg/RN2a436M73) or on [X](https://x.com/harperscarroll)."
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "G9TytWkb3m15"
      },
      "source": [
        "#### Before we begin: A note on OOM errors\n",
        "\n",
        "If you get an error like this: `OutOfMemoryError: CUDA out of memory`, tweak your parameters to make the model less computationally intensive. I will help guide you through that in this guide, and if you have any additional questions you can reach out on the [Discord channel](https://discord.gg/RN2a436M73) or on [X](https://x.com/harperscarroll).\n",
        "\n",
        "To re-try after you tweak your parameters, open a Terminal ('Launcher' or '+' in the nav bar above -> Other -> Terminal) and run the command `nvidia-smi`. Then find the process ID `PID` under `Processes` and run the command `kill [PID]`. You will need to re-start your notebook from the beginning. (There may be a better way to do this... if so please do let me know!)"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "VC-9m2yv3m18"
      },
      "source": [
        "## Let's begin!\n",
        "### 0. Preparing data\n",
        "\n",
        "Before you check out a GPU, prepare your dataset for loading and training.\n",
        "\n",
        "To prepare your dataset for loading, all you need are two `.jsonl` files structured something like this:\n",
        "```\n",
        "{\"input\": \"What color is the sky?\", \"output\": \"The sky is blue.\"}\n",
        "{\"input\": \"Where is the best place to get cloud GPUs?\", \"output\": \"Brev\"}\n",
        "```\n",
        "If you choose to model your data as input/output pairs, you'll want to use something like the second `formatting_func` below, which will will combine all your features into one input string.\n",
        "\n",
        "As you can see below, I have `notes.jsonl` for my `train_dataset` and `notes_validation.jsonl` for my `eval_dataset`.\n",
        "\n",
        "I used Exporter, a free local-only app, to export my Apple Notes to `.txt` files, and then I wrote a script to process each note into one `.jsonl` file. Note that for this script, ChatGPT can help out a LOT if you tell it how your data is currently formatted, how you'd like it to be formatted, and ask it to write a script in a certain language you know well (for any debugging) to do so. I also broke up my journal entries so the training sample vector length was smaller (see the discussion on `max_length` and the data visualization below). I broke it into pieces so that contexts were encapsulated entirely, since I did want the model to understand context about my life. My data were ultimately formatted as:\n",
        "\n",
        "```json\n",
        "{\"note\": \"journal-entry-for-model-to-predict\"}\n",
        "{\"note\": \"journal-entry-for-model-to-predict-1\"}\n",
        "{\"note\": \"journal-entry-for-model-to-predict-2\"}\n",
        "```"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "E2CkxsA43m15"
      },
      "source": [
        "### 1. Instantiate GPU & Load Dataset\n",
        "\n",
        "I used a GPU and dev environment from [brev.dev](https://developer.nvidia.com/brev). The whole thing cost me $1 using a 1xA10G 24GB. Click the badge below to get your preconfigured instance:\n",
        "\n",
        "[![](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/environment/new?instance=A10G:g5.xlarge&diskStorage=256&name=mistral-finetune-own-data&file=https://github.com/brevdev/notebooks/raw/main/mistral-finetune-own-data.ipynb&python=3.10&cuda=12.0.1)\n",
        "\n",
        "A single A10G (as linked) with 24GB GPU Memory was enough for me. You may need more GPUs and/or Memory if your sequence max_length is larger than 512.\n",
        "\n",
        "Once you've checked out your machine and landed in your instance page, select the specs you'd like (I used **Python 3.10 and CUDA 12.0.1**; these should be preconfigured for you if you use the badge above) and click the \"Build\" button to build your verb container. Give this a few minutes.\n",
        "\n",
        "A few minutes after your model has started Running, click the 'Notebook' button on the top right of your screen once it illuminates (you may need to refresh the screen). You will be taken to a Jupyter Lab environment, where you can upload this Notebook.\n",
        "\n",
        "\n",
        "Note: You can connect your cloud credits (AWS or GCP) by clicking \"Org: \" on the top right, and in the panel that slides over, click \"Connect AWS\" or \"Connect GCP\" under \"Connect your cloud\" and follow the instructions linked to attach your credentials."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "FuXIFTFapAMI",
        "outputId": "c8ced1ad-c7b3-44ba-807b-26d7d13906bc"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m23.0.1\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m23.3.2\u001b[0m\n",
            "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\n",
            "\n",
            "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m23.0.1\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m23.3.2\u001b[0m\n",
            "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\n",
            "\n",
            "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m23.0.1\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m23.3.2\u001b[0m\n",
            "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\n",
            "\n",
            "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m23.0.1\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m23.3.2\u001b[0m\n",
            "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\n",
            "\n",
            "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m23.0.1\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m23.3.2\u001b[0m\n",
            "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\n"
          ]
        }
      ],
      "source": [
        "# You only need to run this once per machine\n",
        "!pip install -q -U bitsandbytes\n",
        "!pip install -q -U git+https://github.com/huggingface/transformers.git\n",
        "!pip install -q -U git+https://github.com/huggingface/peft.git\n",
        "!pip install -q -U git+https://github.com/huggingface/accelerate.git\n",
        "!pip install -q -U datasets scipy ipywidgets matplotlib"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "s6f4z8EYmcJ6"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "e464a32bd6a2466ca2e4dffdb844c468",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Generating train split: 0 examples [00:00, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "594369f27da54568a8dce63d739af386",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Generating train split: 0 examples [00:00, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "from datasets import load_dataset\n",
        "\n",
        "train_dataset = load_dataset('json', data_files='notes.jsonl', split='train')\n",
        "eval_dataset = load_dataset('json', data_files='notes_validation.jsonl', split='train')"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "-9KNTJZkyRgn"
      },
      "source": [
        "\n",
        "Let's use Weights & Biases to track our training metrics. You'll need to apply an API key when prompted. Feel free to skip this if you'd like, and just comment out the `wandb` parameters in the `Trainer` definition below."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "DDqUNyIoyRgo"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m23.0.1\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m23.3.2\u001b[0m\n",
            "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "\u001b[34m\u001b[1mwandb\u001b[0m: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: You can find your API key in your browser here: https://wandb.ai/authorize\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: Paste an API key from your profile and hit enter, or press ctrl+c to quit:"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "  ········\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "\u001b[34m\u001b[1mwandb\u001b[0m: Appending key for api.wandb.ai to your netrc file: /home/ubuntu/.netrc\n"
          ]
        }
      ],
      "source": [
        "!pip install -q wandb -U\n",
        "\n",
        "import wandb, os\n",
        "wandb.login()\n",
        "\n",
        "wandb_project = \"journal-finetune\"\n",
        "if len(wandb_project) > 0:\n",
        "    os.environ[\"WANDB_PROJECT\"] = wandb_project"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "uhw8JiOr3m18"
      },
      "source": [
        "### Formatting prompts\n",
        "Then create a `formatting_func` to structure training examples as prompts."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "f-fJR0MlQiTD"
      },
      "outputs": [],
      "source": [
        "def formatting_func(example):\n",
        "    text = f\"### The following is a note by Eevee the Dog: {example['note']}\"\n",
        "    return text"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "sflV0DL2P64_"
      },
      "source": [
        "Here's another common one:\n",
        "\n",
        "```python\n",
        "def formatting_func(example):\n",
        "    text = f\"### Question: {example['input']}\\n ### Answer: {example['output']}\"\n",
        "    return text\n",
        "```"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "shz8Xdv-yRgf"
      },
      "source": [
        "### 2. Load Base Model"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "MJ-5idQwzvg-"
      },
      "source": [
        "Let's now load Mistral - mistralai/Mistral-7B-v0.1 - using 4-bit quantization!"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from huggingface_hub import notebook_login\n",
        "\n",
        "notebook_login()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "colab": {
          "referenced_widgets": [
            "45524c98039a46d5b7745ad7cb638d2f"
          ]
        },
        "id": "E0Nl5mWL0k2T",
        "outputId": "47b6b01d-e9f2-4b70-919c-17ae64993843"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "fbaedb24777e42c6adc61a04e08369f3",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "import torch\n",
        "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
        "\n",
        "base_model_id = \"mistralai/Mistral-7B-v0.1\"\n",
        "bnb_config = BitsAndBytesConfig(\n",
        "    load_in_4bit=True,\n",
        "    bnb_4bit_use_double_quant=True,\n",
        "    bnb_4bit_quant_type=\"nf4\",\n",
        "    bnb_4bit_compute_dtype=torch.bfloat16\n",
        ")\n",
        "\n",
        "model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map=\"auto\")"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "UjNdXolqyRgf"
      },
      "source": [
        "### 3. Tokenization\n",
        "\n",
        "Set up the tokenizer. Add padding on the left as it [makes training use less memory](https://ai.stackexchange.com/questions/41485/while-fine-tuning-a-decoder-only-llm-like-llama-on-chat-dataset-what-kind-of-pa).\n",
        "\n",
        "\n",
        "For `model_max_length`, it's helpful to get a distribution of your data lengths. Let's first tokenize without the truncation/padding, so we can get a length distribution."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "haSUDD9HyRgf",
        "outputId": "22ee95db-2974-4ab0-e0c7-444d04d3e838"
      },
      "outputs": [],
      "source": [
        "tokenizer = AutoTokenizer.from_pretrained(\n",
        "    base_model_id,\n",
        "    padding_side=\"left\",\n",
        "    add_eos_token=True,\n",
        "    add_bos_token=True,\n",
        ")\n",
        "tokenizer.pad_token = tokenizer.eos_token\n",
        "\n",
        "def generate_and_tokenize_prompt(prompt):\n",
        "    return tokenizer(formatting_func(prompt))"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "WHnKLcq4yRgg"
      },
      "source": [
        "Reformat the prompt and tokenize each sample:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "S3iLAwLh3m19"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "ebb7da214be14ba28d1943d75d7ada16",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Map:   0%|          | 0/164 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "d37f65cb91d849ceb16b720dee8927ac",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Map:   0%|          | 0/40 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)\n",
        "tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "O6ewk27p3m19"
      },
      "source": [
        "Let's get a distribution of our dataset lengths, so we can determine the appropriate `max_length` for our input tensors."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "id": "BA8M9yfC3m19",
        "outputId": "99c6d302-9bb6-47b1-cae9-a1cd870b4770"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "204\n"
          ]
        },
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA0kAAAIjCAYAAADWYVDIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAABC4UlEQVR4nO3deVxU9f7H8fcAsggCboA7pORumUuR3HLBSI0yLZdrpVzNFs218tqqpWlWpraobZqlWZZadlNT3MqfmppLWuK+s1QGiCkgfH9/9GA6I2iAI4Pwej4e87jN93znnM+Z74C87/ec79iMMUYAAAAAAEmSm6sLAAAAAICShJAEAAAAABaEJAAAAACwICQBAAAAgAUhCQAAAAAsCEkAAAAAYEFIAgAAAAALQhIAAAAAWBCSAAAAAMCCkASgVBszZoxsNluxHKtt27Zq27at/fmaNWtks9n0+eefF8vx+/Xrp9DQ0GI5VlGlp6drwIABCgkJkc1m07Bhw1xdktMV97j/k2XLlun666+Xt7e3bDabUlJS8u03e/Zs2Ww2HT58uFjruxIKcy6hoaHq16/fFa8JwNWFkATgqpH7h0/uw9vbW9WrV1d0dLSmTZum06dPO+U4J0+e1JgxY7R9+3an7M+ZSnJtBfHSSy9p9uzZeuSRR/TRRx/p/vvvv2jf0NBQ3XHHHcVYXeHMmzdPU6ZMcXUZl/T777+rR48e8vHx0VtvvaWPPvpIvr6+ri6rQH7++WeNGTOmVIQ2AFcfD1cXAACF9cILLygsLExZWVlKTEzUmjVrNGzYME2ePFlfffWVmjVrZu/7zDPP6L///W+h9n/y5EmNHTtWoaGhuv766wv8um+//bZQxymKS9X27rvvKicn54rXcDlWrVqlm266Sc8//7yrS7ls8+bN065du0r0bNjmzZt1+vRpvfjii4qKirpk3/vvv1+9evWSl5dXMVV3aT///LPGjh2rtm3bFnqGtKSdC4CrDyEJwFWnU6dOatmypf356NGjtWrVKt1xxx2688479csvv8jHx0eS5OHhIQ+PK/ur7s8//1T58uXl6el5RY/zT8qVK+fS4xdEcnKyGjVq5Ooyyozk5GRJUmBg4D/2dXd3l7u7+xWuqHiUpnMB4BpcbgegVGjfvr2effZZHTlyRB9//LG9Pb97klasWKHIyEgFBgbKz89P9evX11NPPSXpr/tJWrVqJUmKjY21X9o3e/ZsSX/dd9SkSRNt3bpVt9xyi8qXL29/7YX3JOXKzs7WU089pZCQEPn6+urOO+/UsWPHHPpc7L4I6z7/qbb87kk6c+aMRo4cqVq1asnLy0v169fXq6++KmOMQz+bzabBgwdr8eLFatKkiby8vNS4cWMtW7Ys/zf8AsnJyerfv7+Cg4Pl7e2t6667Th9++KF9e+59OocOHdL//vc/e+3OuJTq448/VosWLeTj46NKlSqpV69eed7f3HH7+eef1a5dO5UvX141atTQpEmT8uzvyJEjuvPOO+Xr66ugoCANHz5cy5cvl81m05o1a+z7+9///qcjR47Yz+XC9z4nJ0fjx49XzZo15e3trQ4dOmj//v0Offbt26fu3bsrJCRE3t7eqlmzpnr16qXU1NR/PO8FCxbYz7tKlSq67777dOLECYdz7tu3rySpVatWstlsl7z3Jr/7eHIvefz+++/VunVreXt765prrtGcOXPyfe26dev00EMPqXLlyvL399cDDzygP/74w6GvzWbTmDFj8hzf+jMwe/Zs3XvvvZKkdu3a2d/j3Pf/n+R3LsYYjRs3TjVr1lT58uXVrl077d69O89rs7KyNHbsWIWHh8vb21uVK1dWZGSkVqxYUaBjAygdmEkCUGrcf//9euqpp/Ttt9/qwQcfzLfP7t27dccdd6hZs2Z64YUX5OXlpf3792v9+vWSpIYNG+qFF17Qc889p4EDB+pf//qXJOnmm2+27+P3339Xp06d1KtXL913330KDg6+ZF3jx4+XzWbTqFGjlJycrClTpigqKkrbt2+3z3gVREFqszLG6M4779Tq1avVv39/XX/99Vq+fLmeeOIJnThxQq+//rpD/++//14LFy7Uo48+qgoVKmjatGnq3r27jh49qsqVK1+0rrNnz6pt27bav3+/Bg8erLCwMC1YsED9+vVTSkqKhg4dqoYNG+qjjz7S8OHDVbNmTY0cOVKSVLVq1QKff37Gjx+vZ599Vj169NCAAQP066+/6o033tAtt9yibdu2Ocyg/PHHH7r99tvVrVs39ejRQ59//rlGjRqlpk2bqlOnTpL+CpXt27dXQkKChg4dqpCQEM2bN0+rV692OO7TTz+t1NRUHT9+3P4++vn5OfSZOHGi3Nzc9Pjjjys1NVWTJk1Snz59tGnTJklSZmam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AwAAAAAKyuUh6aeffpKfn5+8vLz08MMPa9GiRWrUqJESExPl6empwMBAh/7BwcFKTEy86P4mTJiggIAA+6NWrVpX+AwAAAAAlCYuD0n169fX9u3btWnTJj3yyCPq27evfv755yLvb/To0UpNTbU/jh075sRqAQAAAJR2Hq4uwNPTU/Xq1ZMktWjRQps3b9bUqVPVs2dPZWZmKiUlxWE2KSkpSSEhIRfdn5eXl7y8vK502QAAAABKKZfPJF0oJydHGRkZatGihcqVK6e4uDj7tvj4eB09elQREREurBAAAABAaebSmaTRo0erU6dOql27tk6fPq158+ZpzZo1Wr58uQICAtS/f3+NGDFClSpVkr+/vx577DFFREQUeGU7AAAAACgsl4ak5ORkPfDAA0pISFBAQICaNWum5cuXq2PHjpKk119/XW5uburevbsyMjIUHR2tt99+25UlAwAAACjlStz3JDkb35N0cXxPEgAAAMqSq+57kgAAAACgJCAkAQAAAIAFIQkAAAAALAhJAAAAAGBBSAIAAAAAC0ISAAAAAFgQkgAAAADAgpAEAAAAABaEJAAAAACwICQBAAAAgEWRQtLBgwedXQcAAAAAlAhFCkn16tVTu3bt9PHHH+vcuXPOrgkAAAAAXKZIIenHH39Us2bNNGLECIWEhOihhx7SDz/84OzaAAAAAKDYFSkkXX/99Zo6dapOnjypDz74QAkJCYqMjFSTJk00efJk/frrr86uEwAAAACKxWUt3ODh4aFu3bppwYIFevnll7V//349/vjjqlWrlh544AElJCQ4q04AAAAAKBaXFZK2bNmiRx99VNWqVdPkyZP1+OOP68CBA1qxYoVOnjypu+66y1l1AgAAAECx8CjKiyZPnqxZs2YpPj5enTt31pw5c9S5c2e5uf2VucLCwjR79myFhoY6s1YAAAAAuOKKFJKmT5+u//znP+rXr5+qVauWb5+goCC9//77l1UcAAAAABS3IoWkffv2/WMfT09P9e3btyi7BwAAAACXKdI9SbNmzdKCBQvytC9YsEAffvjhZRcFAAAAAK5SpJA0YcIEValSJU97UFCQXnrppcsuCgAAAABcpUgh6ejRowoLC8vTXqdOHR09evSyiwIAAAAAVylSSAoKCtLOnTvztO/YsUOVK1e+7KIAAAAAwFWKFJJ69+6tIUOGaPXq1crO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9keDBg0k/XVfT65mzZo5vK5atWpKTk6WJMXHx6tWrVoOf/S3bt26wDU0btxY7u7u+e47P7/88ovatGnj0NamTZsCz+ZcyqXOM1dERESe5844dkH98ssvqlWrlj0g5VdTv379tH37dtWvX19DhgzRt99+W2z1AcDVjpkkACjj0tPTFRMTo5dffjnPtmrVqtn/u1y5cg7bbDabcnJynFLDldx3cdfi5vbX//9ojLG3/dP9VVfCDTfcoEOHDmnp0qVauXKlevTooaioKIf7pwAA+WMmCQDKuBtuuEG7d+9WaGio6tWr5/Dw9fUt0D7q16+vY8eOKSkpyd62efNmhz6enp6S/rp/6XI1bNhQ69evd2hbv369GjVqdNn7LoiNGzfmed6wYUNJf19WmJCQYN9+4bLnnp6el/U+NGzYUMeOHXM4xoU1SZK/v7969uypd999V59++qm++OILnTp1qsjHBYCygpkkACgjUlNT8/yxnruS3LvvvqvevXvbV3Xbv3+/5s+fr/fee8/hMriL6dixo+rWrau+fftq0qRJOn36tJ555hlJf83ESFJQUJB8fHy0bNky1axZU97e3kVegvuJJ55Qjx491Lx5c0VFRWnJkiVauHChVq5cWaT9FdaCBQvUsmVLRUZGau7cufrhhx/0/vvvS5Lq1aunWrVqacyYMRo/frz27t2r1157zeH1oaGhSk9PV1xcnK677jqVL19e5cuXL/Dxo6KidO2116pv37565ZVXlJaWlmehjMmTJ6tatWpq3ry53NzctGDBAoWEhBT6+5kAoCxiJgkAyog1a9aoefPmDo+xY8eqevXqWr9+vbKzs3XbbbepadOmGjZsmAIDA+2Xjv0Td3d3LV68WOnp6WrVqpUGDBhg/6Pd29tbkuTh4aFp06Zp5syZql69uu66664in0vXrl01depUvfrqq2rcuLFmzpypWbNm5VlW/EoZO3as5s+fr2bNmmnOnDn65JNP7LNY5cqV0yeffKI9e/aoWbNmevnllzVu3DiH19988816+OGH1bNnT1WtWlWTJk0q1PHd3Ny0aNEinT17Vq1bt9aAAQM0fvx4hz4VKlTQpEmT1LJlS7Vq1UqHDx/WN998U+AxBYCyzGasF00DAOAk69evV2RkpPbv36+6deu6uhynsdlsWrRokcP3KwEAShcutwMAOMWiRYvk5+en8PBw7d+/X0OHDlWbNm1KVUACAJQNhCQAgFOcPn1ao0aN0tGjR1WlShVFRUXluRcH+fvuu+8cvuPoQunp6cVYDQCAy+0AAHCxs2fP6sSJExfdXq9evWKsBgBASAIAAAAAC5a4AQAAAAALQhIAAAAAWBCSAAAAAMCCkAQAAAAAFoQkAAAAALAgJAEAAACABSEJAAAAACz+H2NYYVfYwrGOAAAAAElFTkSuQmCC",
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):\n",
        "    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]\n",
        "    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]\n",
        "    print(len(lengths))\n",
        "\n",
        "    # Plotting the histogram\n",
        "    plt.figure(figsize=(10, 6))\n",
        "    plt.hist(lengths, bins=20, alpha=0.7, color='blue')\n",
        "    plt.xlabel('Length of input_ids')\n",
        "    plt.ylabel('Frequency')\n",
        "    plt.title('Distribution of Lengths of input_ids')\n",
        "    plt.show()\n",
        "\n",
        "plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "nBk4Qp_vyRgh"
      },
      "source": [
        "From here, you can choose where you'd like to set the `max_length` to be. You can truncate and pad training examples to fit them to your chosen size. Be aware that choosing a larger `max_length` has its compute tradeoffs.\n",
        "\n",
        "I'm using my personal notes to train the model, and they vary greatly in length. I spent some time cleaning the dataset so the samples were about the same length, cutting up individual notes if needed, but being sure to not cut in the middle of a word or sentence."
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "bMlw8h743m19"
      },
      "source": [
        "Now let's tokenize again with padding and truncation, and set up the tokenize function to make labels and input_ids the same. This is basically what [self-supervised fine-tuning is](https://neptune.ai/blog/self-supervised-learning)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "id": "acINaViR3m19"
      },
      "outputs": [],
      "source": [
        "max_length = 512 # This was an appropriate max length for my dataset\n",
        "\n",
        "def generate_and_tokenize_prompt2(prompt):\n",
        "    result = tokenizer(\n",
        "        formatting_func(prompt),\n",
        "        truncation=True,\n",
        "        max_length=max_length,\n",
        "        padding=\"max_length\",\n",
        "    )\n",
        "    result[\"labels\"] = result[\"input_ids\"].copy()\n",
        "    return result"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "colab": {
          "referenced_widgets": [
            "518d4f0b89bf4d57bf00d4c6d6e59eb5"
          ]
        },
        "id": "lTk-aTog3m19",
        "outputId": "4fb637b4-77a2-47c6-de7b-4fb620663dd7"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "bb016cb35d734a2c85a5649b2e84711c",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Map:   0%|          | 0/164 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "c0cb00d65a474115960fa7b79c0b164b",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Map:   0%|          | 0/40 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)\n",
        "tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt2)"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "TQL796OayRgh"
      },
      "source": [
        "Check that `input_ids` is padded on the left with the `eos_token` (2) and there is an `eos_token` 2 added to the end, and the prompt starts with a `bos_token` (1)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "OKHhvxK83m19"
      },
      "outputs": [],
      "source": [
        "print(tokenized_train_dataset[1]['input_ids'])"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "I6LRa2Zm3m19"
      },
      "source": [
        "Now all the samples should be the same length, `max_length`."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "id": "I55Yo3yy3m19",
        "outputId": "c87e344d-e0f3-4542-afcc-4e2025926d64"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "204\n"
          ]
        },
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA1IAAAIjCAYAAAAJLyrXAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAABLkklEQVR4nO3deXwN9/7H8feRyL4JIgmRxL6rvSpVaqdUpbWUFqW62CrV+ulmaV2tolSVbqRaumjR0ktrCVpFLQ3lEqL2JLhViShJJPP7o4+c2yMRmUhysryej8c8buc735n5zMkI7/ud+R6LYRiGAAAAAAC5VsbeBQAAAABAcUOQAgAAAACTCFIAAAAAYBJBCgAAAABMIkgBAAAAgEkEKQAAAAAwiSAFAAAAACYRpAAAAADAJIIUAAAAAJhEkAJQ6k2ePFkWi6VQztWuXTu1a9fOur5582ZZLBZ99dVXhXL+IUOGKCQkpFDOlVfJyckaPny4/P39ZbFY9Mwzz9i7pHxX2D/3W1m3bp3uuOMOubi4yGKx6NKlS9n2i4yMlMVi0YkTJwq1voJg5lpCQkI0ZMiQAq8JQPFCkAJQomT+4yhzcXFxUWBgoLp06aK3335bly9fzpfzxMXFafLkyYqOjs6X4+WnolxbbvzrX/9SZGSknnrqKX3yySd65JFHbto3JCRE9913XyFWZ86yZcs0Z84ce5eRoz/++EN9+/aVq6ur5s+fr08++UTu7u72LitX/vOf/2jy5MklItgBKH4c7V0AABSEqVOnKjQ0VGlpaUpISNDmzZv1zDPPaPbs2fr222/VqFEja9+XXnpJ//d//2fq+HFxcZoyZYpCQkJ0xx135Hq/H374wdR58iKn2j744ANlZGQUeA23Y9OmTbrzzjs1adIke5dy25YtW6YDBw4U6VG1Xbt26fLly3r11VfVsWPHHPs+8sgj6t+/v5ydnQupupz95z//0ZQpU9SuXTvTI61F7VoAFD8EKQAlUrdu3dS8eXPr+sSJE7Vp0ybdd9996tWrlw4dOiRXV1dJkqOjoxwdC/bX4V9//SU3Nzc5OTkV6HlupWzZsnY9f26cP39e9erVs3cZpcb58+clST4+Prfs6+DgIAcHhwKuqHCUpGsBYB882geg1Lj33nv18ssv6+TJk/r000+t7dm9I7V+/XqFhYXJx8dHHh4eql27tl544QVJf7/f0qJFC0nS0KFDrY8RRkZGSvr7PagGDRpoz549atu2rdzc3Kz73viOVKb09HS98MIL8vf3l7u7u3r16qXTp0/b9LnZexr/POatasvuHakrV67o2WefVVBQkJydnVW7dm3NnDlThmHY9LNYLBo1apRWrVqlBg0ayNnZWfXr19e6deuy/8BvcP78eQ0bNkyVKlWSi4uLGjdurI8//ti6PfO9oePHj+u7776z1p4fj219+umnatasmVxdXeXr66v+/ftn+Xwzf27/+c9/1L59e7m5ualy5cqaMWNGluOdPHlSvXr1kru7u/z8/DRu3Dh9//33slgs2rx5s/V43333nU6ePGm9lhs/+4yMDE2bNk1VqlSRi4uLOnTooNjYWJs+R48eVXh4uPz9/eXi4qIqVaqof//+SkxMvOV1L1++3HrdFSpU0KBBg3T27Fmbax48eLAkqUWLFrJYLDm+C5Tde0WZj1f+9NNPatmypVxcXFStWjUtWbIk2323bt2qJ554QuXLl5eXl5ceffRR/fnnnzZ9LRaLJk+enOX8//wzEBkZqYceekiS1L59e+tnnPn530p212IYhl577TVVqVJFbm5uat++vQ4ePJhl37S0NE2ZMkU1a9aUi4uLypcvr7CwMK1fvz5X5wZQMjAiBaBUeeSRR/TCCy/ohx9+0OOPP55tn4MHD+q+++5To0aNNHXqVDk7Oys2Nlbbtm2TJNWtW1dTp07VK6+8ohEjRujuu++WJN11113WY/zxxx/q1q2b+vfvr0GDBqlSpUo51jVt2jRZLBZNmDBB58+f15w5c9SxY0dFR0dbR85yIze1/ZNhGOrVq5eioqI0bNgw3XHHHfr+++/13HPP6ezZs3rrrbds+v/0009asWKFnn76aXl6eurtt99WeHi4Tp06pfLly9+0rqtXr6pdu3aKjY3VqFGjFBoaquXLl2vIkCG6dOmSxo4dq7p16+qTTz7RuHHjVKVKFT377LOSpIoVK+b6+rMzbdo0vfzyy+rbt6+GDx+uCxcuaN68eWrbtq1+/fVXm5GYP//8U127dlWfPn3Ut29fffXVV5owYYIaNmyobt26Sfo7eN57772Kj4/X2LFj5e/vr2XLlikqKsrmvC+++KISExN15swZ6+fo4eFh0+f1119XmTJlNH78eCUmJmrGjBkaOHCgdu7cKUlKTU1Vly5dlJKSotGjR8vf319nz57VmjVrdOnSJXl7e9/0uiMjIzV06FC1aNFC06dP17lz5zR37lxt27bNet0vvviiateurffff9/6OGz16tVNf8axsbF68MEHNWzYMA0ePFiLFi3SkCFD1KxZM9WvX9+m76hRo+Tj46PJkycrJiZGCxYs0MmTJ61BOrfatm2rMWPG6O2339YLL7ygunXrSpL1f/PilVde0Wuvvabu3bure/fu2rt3rzp37qzU1FSbfpMnT9b06dM1fPhwtWzZUklJSdq9e7f27t2rTp065fn8AIoZAwBKkMWLFxuSjF27dt20j7e3t9GkSRPr+qRJk4x//jp86623DEnGhQsXbnqMXbt2GZKMxYsXZ9l2zz33GJKMhQsXZrvtnnvusa5HRUUZkozKlSsbSUlJ1vYvv/zSkGTMnTvX2hYcHGwMHjz4lsfMqbbBgwcbwcHB1vVVq1YZkozXXnvNpt+DDz5oWCwWIzY21tomyXBycrJp27dvnyHJmDdvXpZz/dOcOXMMScann35qbUtNTTVat25teHh42Fx7cHCw0aNHjxyPl9u+J06cMBwcHIxp06bZtP/222+Go6OjTXvmz23JkiXWtpSUFMPf398IDw+3ts2aNcuQZKxatcradvXqVaNOnTqGJCMqKsra3qNHD5vPO1Pmz71u3bpGSkqKtX3u3LmGJOO3334zDMMwfv31V0OSsXz58lt/GP+Qmppq+Pn5GQ0aNDCuXr1qbV+zZo0hyXjllVesbbn5M3Nj3+PHj1vbgoODDUnG1q1brW3nz583nJ2djWeffTbLvs2aNTNSU1Ot7TNmzDAkGd988421TZIxadKkLOe/8c/A8uXLs3zmuXXjtZw/f95wcnIyevToYWRkZFj7vfDCC4Ykm/M2btw41/cogJKLR/sAlDoeHh45zt6XOULxzTff5HliBmdnZw0dOjTX/R999FF5enpa1x988EEFBATo3//+d57On1v//ve/5eDgoDFjxti0P/vsszIMQ2vXrrVp79ixo82IRaNGjeTl5aXff//9lufx9/fXgAEDrG1ly5bVmDFjlJycrC1btuTD1WS1YsUKZWRkqG/fvvrvf/9rXfz9/VWzZs0so0geHh4aNGiQdd3JyUktW7a0ub5169apcuXK6tWrl7XNxcXlpiOcORk6dKjNe3OZI4iZ58sccfr+++/1119/5fq4u3fv1vnz5/X000/LxcXF2t6jRw/VqVNH3333nelac1KvXj1r7dLfo4i1a9fO9r4YMWKEzbt6Tz31lBwdHQv8Xr+VDRs2KDU1VaNHj7YZGctuohAfHx8dPHhQR48eLcQKARQ1BCkApU5ycrJNaLlRv3791KZNGw0fPlyVKlVS//799eWXX5oKVZUrVzY1sUTNmjVt1i0Wi2rUqFHg0zqfPHlSgYGBWT6PzMejTp48adNetWrVLMcoV65clndcsjtPzZo1VaaM7V87NztPfjl69KgMw1DNmjVVsWJFm+XQoUPWiRYyValSJcvjZTde38mTJ1W9evUs/WrUqGG6vhs/z3LlykmS9XyhoaGKiIjQhx9+qAoVKqhLly6aP3/+Ld+Pyvw8a9eunWVbnTp18v3zNnNf3Hive3h4KCAgwO5TmGd+JjfWV7FiRevPJdPUqVN16dIl1apVSw0bNtRzzz2n/fv3F1qtAIoGghSAUuXMmTNKTEzM8R+9rq6u2rp1qzZs2KBHHnlE+/fvV79+/dSpUyelp6fn6jxm3mvKrZu9P5LbmvLDzWY5M26YmKKoyMjIkMVi0bp167R+/fosy3vvvWfTv7CvLzfnmzVrlvbv368XXnhBV69e1ZgxY1S/fn2dOXOmQGrKi8L63ArzXs9J27ZtdezYMS1atEgNGjTQhx9+qKZNm+rDDz+0d2kAChFBCkCp8sknn0iSunTpkmO/MmXKqEOHDpo9e7b+85//aNq0adq0aZP1UTAzL8Xnxo2PCBmGodjYWJtZ3sqVK6dLly5l2ffG0QUztQUHBysuLi7Lo46HDx+2bs8PwcHBOnr0aJZRvfw+z42qV68uwzAUGhqqjh07ZlnuvPNO08cMDg7WsWPHsoSEG2fbk/LvPmnYsKFeeuklbd26VT/++KPOnj2rhQsX5lijJMXExGTZFhMTU2Cfd27ceK8nJycrPj7+lvd6amqq4uPjbdry889h5mdyY30XLlzIdmTN19dXQ4cO1WeffabTp0+rUaNG2c40CKDkIkgBKDU2bdqkV199VaGhoRo4cOBN+128eDFLW+YX26akpEiS3N3dJSnbYJMXS5YssQkzX331leLj460zxUl/h4IdO3bYzCC2Zs2aLNN4m6mte/fuSk9P1zvvvGPT/tZbb8lisdic/3Z0795dCQkJ+uKLL6xt169f17x58+Th4aF77rknX85zoz59+sjBwUFTpkzJEnwMw9Aff/xh+phdunTR2bNn9e2331rbrl27pg8++CBLX3d391xNU34zSUlJun79uk1bw4YNVaZMGeu9mJ3mzZvLz89PCxcutOm3du1aHTp0SD169MhzTbfr/fffV1pamnV9wYIFun79epZ7fevWrVn2u3FEKj//HHbs2FFly5bVvHnzbO6VOXPmZOl7433j4eGhGjVq5PgzAVDyMP05gBJp7dq1Onz4sK5fv65z585p06ZNWr9+vYKDg/Xtt9/avIB/o6lTp2rr1q3q0aOHgoODdf78eb377ruqUqWKwsLCJP39Dz0fHx8tXLhQnp6ecnd3V6tWrRQaGpqnen19fRUWFqahQ4fq3LlzmjNnjmrUqGEzgcHw4cP11VdfqWvXrurbt6+OHTumTz/9NMt01WZq69mzp9q3b68XX3xRJ06cUOPGjfXDDz/om2++0TPPPJOnqbCzM2LECL333nsaMmSI9uzZo5CQEH311Vfatm2b5syZk+M7a7cSGxur1157LUt7kyZN1KNHD7322muaOHGiTpw4od69e8vT01PHjx/XypUrNWLECI0fP97U+Z544gm98847GjBggMaOHauAgAAtXbrUek/9c5SkWbNm+uKLLxQREaEWLVrIw8NDPXv2zPW5Nm3apFGjRumhhx5SrVq1dP36dX3yySdycHBQeHj4TfcrW7as3njjDQ0dOlT33HOPBgwYYJ3+PCQkROPGjTN1zfkpNTVVHTp0UN++fRUTE6N3331XYWFhNpN3DB8+XE8++aTCw8PVqVMn7du3T99//70qVKhgc6w77rhDDg4OeuONN5SYmChnZ2fde++98vPzM11XxYoVNX78eE2fPl333Xefunfvrl9//VVr167Nct569eqpXbt2atasmXx9fbV792599dVXGjVqVN4+FADFk30mCwSAgpE5pXHm4uTkZPj7+xudOnUy5s6dazPNdqYbpz/fuHGjcf/99xuBgYGGk5OTERgYaAwYMMA4cuSIzX7ffPONUa9ePcPR0dFmuvF77rnHqF+/frb13Wz6888++8yYOHGi4efnZ7i6uho9evQwTp48mWX/WbNmGZUrVzacnZ2NNm3aGLt3785yzJxqu3H6c8MwjMuXLxvjxo0zAgMDjbJlyxo1a9Y03nzzTZspoA3j7ympR44cmaWmm03L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sWPH6tlnn9WBAwf0xBNPaOjQoYqKipL092NwXbt2Vd++fRUfH6+5c+fm+vPI6TozvfzyywoPD9e+ffs0cOBA9e/fX4cOHcrV8e+66y7NmTNHXl5eio+PV3x8vMaPH5/r+iQpIyNDffr0kZOTk3bu3KmFCxdqwoQJNn3efvttffvtt/ryyy8VExOjpUuXKiQkxNR5AKC04tE+AMAtvfPOO2rSpIn+9a9/WdsWLVqkoKAgHTlyRLVq1ZIkNWrUSJMmTZIk1axZU++88442btyoTp06af369Tp27Jg2b94sf39/SdK0adPUqVMn6zEzH00rX768tU+mcuXK6Z133pGDg4Pq1KmjHj16aOPGjXr88cezrXnmzJkaMmSInn76aUlSRESEduzYoZkzZ6p9+/aqWLGinJ2d5erqmuVct5LTdWZ66KGHNHz4cEnSq6++qvXr12vevHl69913b3l8JycneXt7y2KxmK4t04YNG3T48GF9//33CgwMlCT961//Urdu3ax9Tp06pZo1ayosLEwWi0XBwcF5OhcAlEaMSAEAbmnfvn2KioqSh4eHdalTp46kv98zytSoUSOb/QICAnT+/HlJUkxMjIKCgmyCQcuWLXNdQ/369eXg4JDtsbNz6NAhtWnTxqatTZs2uR4VyklO15mpdevWWdbz49y5dejQIQUFBVlDVHY1DRkyRNHR0apdu7bGjBmjH374odDqA4DijhEpAMAtJScnq2fPnnrjjTeybAsICLD+d9myZW22WSwWZWRk5EsNBXnswq6lTJm//39MwzCsbbd636sgNG3aVMePH9fatWu1YcMG9e3bVx07drR5nwsAkD1GpAAAt9S0aVMdPHhQISEhqlGjhs3i7u6eq2PUrl1bp0+f1rlz56xtu3btsunj5OQk6e/3qW5X3bp1tW3bNpu2bdu2qV69erd97NzYsWNHlvW6detK+t8jjPHx8dbtN0757uTkdFufQ926dXX69Gmbc9xYkyR5eXmpX79++uCDD/TFF1/o66+/1sWLF/N8XgAoLRiRAgBYJSYmZvkHfeYMeR988IEGDBhgna0uNjZWn3/+uT788EObR+5uplOnTqpevboGDx6sGTNm6PLly3rppZck/T2iI0l+fn5ydXXVunXrVKVKFbm4uOR5+vHnnntOffv2VZMmTdSxY0etXr1aK1as0IYNG/J0PLOWL1+u5s2bKywsTEuXLtUvv/yijz76SJJUo0YNBQUFafLkyZo2bZqOHDmiWbNm2ewfEhKi5ORkbdy4UY0bN5abm5vc3Nxyff6OHTuqVq1aGjx4sN58800lJSVlmdxj9uzZCggIUJMmTVSmTBktX75c/v7+pr+/CgBKI0akAABWmzdvVpMmTWyWKVOmKDAwUNu2bVN6ero6d+6shg0b6plnnpGPj4/1MbVbcXBw0KpVq5ScnKwWLVpo+PDh1n/Yu7i4SJIcHR319ttv67333lNgYKDuv//+PF9L7969NXfuXM2cOVP169fXe++9p8WLF2eZUr2gTJkyRZ9//rkaNWqkJUuW6LPPPrOOhpUtW1afffaZDh8+rEaNGumNN97Qa6+9ZrP/XXfdpSeffFL9+vVTxYoVNWPGDFPnL1OmjFauXKmrV6+qZcuWGj58uKZNm2bTx9PTUzNmzFDz5s3VokULnThxQv/+979z/TMFgNLMYvzzAW0AAArRtm3bFBYWptjYWFWvXt3e5eQbi8WilStX2nz/FACgZOHRPgBAoVm5cqU8PDxUs2ZNxcbGauzYsWrTpk2JClEAgNKBIAUAKDSXL1/WhAkTdOrUKVWoUEEdO3bM8m4Qsvfjjz/afAfUjZKTkwuxGgAAj/YBAFAMXL16VWfPnr3p9ho1ahRiNQAAghQAAAAAmMS0PAAAAABgEkEKAAAAAEwiSAEAAACASQQpAAAAADCJIAUAAAAAJhGkAAAAAMAkghQAAAAAmPT/sqs4+kF7nacAAAAASUVORK5CYII=",
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "jP3R4enP3m19"
      },
      "source": [
        "### How does the base model do?"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "Vxbl4ACsyRgi"
      },
      "source": [
        "Optionally, you can check how Mistral does on one of your data samples. For example, if you have a dataset of users' biometric data to their health scores, you could test the following `eval_prompt`:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "id": "gOxnx-cAyRgi"
      },
      "outputs": [],
      "source": [
        "eval_prompt = \"\"\" Given the following biometric data, score the users' health, from 0-100.\n",
        "\n",
        "### Biometric Data:\n",
        "Temperature=98.2,\n",
        "Sex=F,\n",
        "Age=29,\n",
        "Height=69 inches,\n",
        "Weight=160 lbs,\n",
        "V02_Max=55,\n",
        "HRV=55\n",
        "\n",
        "### Health Score:\n",
        "\"\"\""
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "KRhfq_Fa3m19"
      },
      "source": [
        "The `eval_prompt` I used was:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "id": "pa6ux9ni3m19"
      },
      "outputs": [],
      "source": [
        "eval_prompt = \" The following is a note by Eevee the Dog: # \""
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "NidIuFXMyRgi",
        "outputId": "b1794b11-9a22-4b0a-e871-7df039ab59fc"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            " The following is a note by Eevee the Dog: # 10\n",
            "\n",
            "I am not sure what to make of this. I have been told that it is a “selfie” but I don’t know what that means. It seems like a picture of me, so I guess it must be one. But why would anyone want to take my picture? And why do they call it a selfie? I think it is just another way for people to get into my head and steal my thoughts. They are always trying to figure out what makes me tick. Well, I will tell you what makes me tick. My tail does!\n"
          ]
        }
      ],
      "source": [
        "# Init an eval tokenizer that doesn't add padding or eos token\n",
        "eval_tokenizer = AutoTokenizer.from_pretrained(\n",
        "    base_model_id,\n",
        "    add_bos_token=True,\n",
        ")\n",
        "\n",
        "model_input = eval_tokenizer(eval_prompt, return_tensors=\"pt\").to(\"cuda\")\n",
        "\n",
        "model.eval()\n",
        "with torch.no_grad():\n",
        "    print(eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.15)[0], skip_special_tokens=True))"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "dCAWeCzZyRgi"
      },
      "source": [
        "Observe how the model does out of the box."
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "AapDoyfAyRgi"
      },
      "source": [
        "### 4. Set Up LoRA"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "Mp2gMi1ZzGET"
      },
      "source": [
        "Now, to start our fine-tuning, we have to apply some preprocessing to the model to prepare it for training. For that use the `prepare_model_for_kbit_training` method from PEFT."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "id": "a9EUEDAl0ss3"
      },
      "outputs": [],
      "source": [
        "from peft import prepare_model_for_kbit_training\n",
        "\n",
        "model.gradient_checkpointing_enable()\n",
        "model = prepare_model_for_kbit_training(model)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "id": "gkIcwsSU01EB"
      },
      "outputs": [],
      "source": [
        "def print_trainable_parameters(model):\n",
        "    \"\"\"\n",
        "    Prints the number of trainable parameters in the model.\n",
        "    \"\"\"\n",
        "    trainable_params = 0\n",
        "    all_param = 0\n",
        "    for _, param in model.named_parameters():\n",
        "        all_param += param.numel()\n",
        "        if param.requires_grad:\n",
        "            trainable_params += param.numel()\n",
        "    print(\n",
        "        f\"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}\"\n",
        "    )"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "cUYEpEK-yRgj"
      },
      "source": [
        "Let's print the model to examine its layers, as we will apply QLoRA to all the linear layers of the model. Those layers are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, and `lm_head`."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "metadata": {
        "id": "XshGNsbxyRgj",
        "outputId": "c619b0e8-8516-4d4b-9abe-13eaa3f3b204",
        "scrolled": true
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "MistralForCausalLM(\n",
            "  (model): MistralModel(\n",
            "    (embed_tokens): Embedding(32000, 4096)\n",
            "    (layers): ModuleList(\n",
            "      (0-31): 32 x MistralDecoderLayer(\n",
            "        (self_attn): MistralSdpaAttention(\n",
            "          (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)\n",
            "          (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)\n",
            "          (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)\n",
            "          (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)\n",
            "          (rotary_emb): MistralRotaryEmbedding()\n",
            "        )\n",
            "        (mlp): MistralMLP(\n",
            "          (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)\n",
            "          (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)\n",
            "          (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)\n",
            "          (act_fn): SiLU()\n",
            "        )\n",
            "        (input_layernorm): MistralRMSNorm()\n",
            "        (post_attention_layernorm): MistralRMSNorm()\n",
            "      )\n",
            "    )\n",
            "    (norm): MistralRMSNorm()\n",
            "  )\n",
            "  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)\n",
            ")\n"
          ]
        }
      ],
      "source": [
        "print(model)"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "I6mTLuQJyRgj"
      },
      "source": [
        "Here we define the LoRA config.\n",
        "\n",
        "`r` is the rank of the low-rank matrix used in the adapters, which thus controls the number of parameters trained. A higher rank will allow for more expressivity, but there is a compute tradeoff.\n",
        "\n",
        "`alpha` is the scaling factor for the learned weights. The weight matrix is scaled by `alpha/r`, and thus a higher value for `alpha` assigns more weight to the LoRA activations.\n",
        "\n",
        "The values used in the QLoRA paper were `r=64` and `lora_alpha=16`, and these are said to generalize well, but we will use `r=32` and `lora_alpha=64` so that we have more emphasis on the new fine-tuned data while also reducing computational complexity."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "metadata": {
        "id": "Ybeyl20n3dYH",
        "outputId": "6a16c182-04d9-4812-ae81-502a8fe364d0"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "trainable params: 85041152 || all params: 3837112320 || trainable%: 2.2162799758751914\n"
          ]
        }
      ],
      "source": [
        "from peft import LoraConfig, get_peft_model\n",
        "\n",
        "config = LoraConfig(\n",
        "    r=32,\n",
        "    lora_alpha=64,\n",
        "    target_modules=[\n",
        "        \"q_proj\",\n",
        "        \"k_proj\",\n",
        "        \"v_proj\",\n",
        "        \"o_proj\",\n",
        "        \"gate_proj\",\n",
        "        \"up_proj\",\n",
        "        \"down_proj\",\n",
        "        \"lm_head\",\n",
        "    ],\n",
        "    bias=\"none\",\n",
        "    lora_dropout=0.05,  # Conventional\n",
        "    task_type=\"CAUSAL_LM\",\n",
        ")\n",
        "\n",
        "model = get_peft_model(model, config)\n",
        "print_trainable_parameters(model)"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "X_FHi_VLyRgn"
      },
      "source": [
        "See how the model looks different now, with the LoRA adapters added:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "id": "IaYMWak4yRgn"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "PeftModelForCausalLM(\n",
            "  (base_model): LoraModel(\n",
            "    (model): MistralForCausalLM(\n",
            "      (model): MistralModel(\n",
            "        (embed_tokens): Embedding(32000, 4096)\n",
            "        (layers): ModuleList(\n",
            "          (0-31): 32 x MistralDecoderLayer(\n",
            "            (self_attn): MistralSdpaAttention(\n",
            "              (q_proj): lora.Linear4bit(\n",
            "                (base_layer): Linear4bit(in_features=4096, out_features=4096, bias=False)\n",
            "                (lora_dropout): ModuleDict(\n",
            "                  (default): Dropout(p=0.05, inplace=False)\n",
            "                )\n",
            "                (lora_A): ModuleDict(\n",
            "                  (default): Linear(in_features=4096, out_features=32, bias=False)\n",
            "                )\n",
            "                (lora_B): ModuleDict(\n",
            "                  (default): Linear(in_features=32, out_features=4096, bias=False)\n",
            "                )\n",
            "                (lora_embedding_A): ParameterDict()\n",
            "                (lora_embedding_B): ParameterDict()\n",
            "              )\n",
            "              (k_proj): lora.Linear4bit(\n",
            "                (base_layer): Linear4bit(in_features=4096, out_features=1024, bias=False)\n",
            "                (lora_dropout): ModuleDict(\n",
            "                  (default): Dropout(p=0.05, inplace=False)\n",
            "                )\n",
            "                (lora_A): ModuleDict(\n",
            "                  (default): Linear(in_features=4096, out_features=32, bias=False)\n",
            "                )\n",
            "                (lora_B): ModuleDict(\n",
            "                  (default): Linear(in_features=32, out_features=1024, bias=False)\n",
            "                )\n",
            "                (lora_embedding_A): ParameterDict()\n",
            "                (lora_embedding_B): ParameterDict()\n",
            "              )\n",
            "              (v_proj): lora.Linear4bit(\n",
            "                (base_layer): Linear4bit(in_features=4096, out_features=1024, bias=False)\n",
            "                (lora_dropout): ModuleDict(\n",
            "                  (default): Dropout(p=0.05, inplace=False)\n",
            "                )\n",
            "                (lora_A): ModuleDict(\n",
            "                  (default): Linear(in_features=4096, out_features=32, bias=False)\n",
            "                )\n",
            "                (lora_B): ModuleDict(\n",
            "                  (default): Linear(in_features=32, out_features=1024, bias=False)\n",
            "                )\n",
            "                (lora_embedding_A): ParameterDict()\n",
            "                (lora_embedding_B): ParameterDict()\n",
            "              )\n",
            "              (o_proj): lora.Linear4bit(\n",
            "                (base_layer): Linear4bit(in_features=4096, out_features=4096, bias=False)\n",
            "                (lora_dropout): ModuleDict(\n",
            "                  (default): Dropout(p=0.05, inplace=False)\n",
            "                )\n",
            "                (lora_A): ModuleDict(\n",
            "                  (default): Linear(in_features=4096, out_features=32, bias=False)\n",
            "                )\n",
            "                (lora_B): ModuleDict(\n",
            "                  (default): Linear(in_features=32, out_features=4096, bias=False)\n",
            "                )\n",
            "                (lora_embedding_A): ParameterDict()\n",
            "                (lora_embedding_B): ParameterDict()\n",
            "              )\n",
            "              (rotary_emb): MistralRotaryEmbedding()\n",
            "            )\n",
            "            (mlp): MistralMLP(\n",
            "              (gate_proj): lora.Linear4bit(\n",
            "                (base_layer): Linear4bit(in_features=4096, out_features=14336, bias=False)\n",
            "                (lora_dropout): ModuleDict(\n",
            "                  (default): Dropout(p=0.05, inplace=False)\n",
            "                )\n",
            "                (lora_A): ModuleDict(\n",
            "                  (default): Linear(in_features=4096, out_features=32, bias=False)\n",
            "                )\n",
            "                (lora_B): ModuleDict(\n",
            "                  (default): Linear(in_features=32, out_features=14336, bias=False)\n",
            "                )\n",
            "                (lora_embedding_A): ParameterDict()\n",
            "                (lora_embedding_B): ParameterDict()\n",
            "              )\n",
            "              (up_proj): lora.Linear4bit(\n",
            "                (base_layer): Linear4bit(in_features=4096, out_features=14336, bias=False)\n",
            "                (lora_dropout): ModuleDict(\n",
            "                  (default): Dropout(p=0.05, inplace=False)\n",
            "                )\n",
            "                (lora_A): ModuleDict(\n",
            "                  (default): Linear(in_features=4096, out_features=32, bias=False)\n",
            "                )\n",
            "                (lora_B): ModuleDict(\n",
            "                  (default): Linear(in_features=32, out_features=14336, bias=False)\n",
            "                )\n",
            "                (lora_embedding_A): ParameterDict()\n",
            "                (lora_embedding_B): ParameterDict()\n",
            "              )\n",
            "              (down_proj): lora.Linear4bit(\n",
            "                (base_layer): Linear4bit(in_features=14336, out_features=4096, bias=False)\n",
            "                (lora_dropout): ModuleDict(\n",
            "                  (default): Dropout(p=0.05, inplace=False)\n",
            "                )\n",
            "                (lora_A): ModuleDict(\n",
            "                  (default): Linear(in_features=14336, out_features=32, bias=False)\n",
            "                )\n",
            "                (lora_B): ModuleDict(\n",
            "                  (default): Linear(in_features=32, out_features=4096, bias=False)\n",
            "                )\n",
            "                (lora_embedding_A): ParameterDict()\n",
            "                (lora_embedding_B): ParameterDict()\n",
            "              )\n",
            "              (act_fn): SiLU()\n",
            "            )\n",
            "            (input_layernorm): MistralRMSNorm()\n",
            "            (post_attention_layernorm): MistralRMSNorm()\n",
            "          )\n",
            "        )\n",
            "        (norm): MistralRMSNorm()\n",
            "      )\n",
            "      (lm_head): lora.Linear(\n",
            "        (base_layer): Linear(in_features=4096, out_features=32000, bias=False)\n",
            "        (lora_dropout): ModuleDict(\n",
            "          (default): Dropout(p=0.05, inplace=False)\n",
            "        )\n",
            "        (lora_A): ModuleDict(\n",
            "          (default): Linear(in_features=4096, out_features=32, bias=False)\n",
            "        )\n",
            "        (lora_B): ModuleDict(\n",
            "          (default): Linear(in_features=32, out_features=32000, bias=False)\n",
            "        )\n",
            "        (lora_embedding_A): ParameterDict()\n",
            "        (lora_embedding_B): ParameterDict()\n",
            "      )\n",
            "    )\n",
            "  )\n",
            ")\n"
          ]
        }
      ],
      "source": [
        "print(model)"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "_0MOtwf3zdZp"
      },
      "source": [
        "### 5. Run Training!"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "fEe0uWYSyRgo"
      },
      "source": [
        "I didn't have a lot of training samples: only about 200 total train/validation. I used 500 training steps, and I was fine with overfitting in this case. I found that the end product worked well. It took about 20 minutes on the 1x A10G 24GB.\n",
        "\n",
        "Overfitting is when the validation loss goes up (bad) while the training loss goes down significantly, meaning the model is learning the training set really well, but is unable to generalize to new datapoints. In most cases, this is not desired, but since I am just playing around with a model to generate outputs like my journal entries, I was fine with a moderate amount of overfitting.\n",
        "\n",
        "With that said, a note on training: you can set the `max_steps` to be high initially, and examine at what step your model's performance starts to degrade. There is where you'll find a sweet spot for how many steps to perform. For example, say you start with 1000 steps, and find that at around 500 steps the model starts overfitting, as described above. Therefore, 500 steps would be your sweet spot, so you would use the `checkpoint-500` model repo in your output dir (`mistral-journal-finetune`) as your final model in step 6 below.\n",
        "\n",
        "If you're just doing something for fun like I did and are OK with overfitting, you can try different checkpoint versions with different degrees of overfitting.\n",
        "\n",
        "You can interrupt the process via Kernel -> Interrupt Kernel in the top nav bar once you realize you didn't need to train anymore."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "metadata": {
        "id": "c_L1131GyRgo"
      },
      "outputs": [],
      "source": [
        "if torch.cuda.device_count() > 1: # If more than 1 GPU\n",
        "    model.is_parallelizable = True\n",
        "    model.model_parallel = True"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 26,
      "metadata": {
        "id": "yxSbpKQSLY6B"
      },
      "outputs": [],
      "source": [
        "model = accelerator.prepare_model(model)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jq0nX33BmfaC"
      },
      "outputs": [],
      "source": [
        "import transformers\n",
        "from datetime import datetime\n",
        "\n",
        "project = \"journal-finetune\"\n",
        "base_model_name = \"mistral\"\n",
        "run_name = base_model_name + \"-\" + project\n",
        "output_dir = \"./\" + run_name\n",
        "\n",
        "trainer = transformers.Trainer(\n",
        "    model=model,\n",
        "    train_dataset=tokenized_train_dataset,\n",
        "    eval_dataset=tokenized_val_dataset,\n",
        "    args=transformers.TrainingArguments(\n",
        "        output_dir=output_dir,\n",
        "        warmup_steps=2,\n",
        "        per_device_train_batch_size=2,\n",
        "        gradient_accumulation_steps=1,\n",
        "        gradient_checkpointing=True,\n",
        "        max_steps=500,\n",
        "        learning_rate=2.5e-5, # Want a small lr for finetuning\n",
        "        bf16=True,\n",
        "        optim=\"paged_adamw_8bit\",\n",
        "        logging_steps=25,              # When to start reporting loss\n",
        "        logging_dir=\"./logs\",        # Directory for storing logs\n",
        "        save_strategy=\"steps\",       # Save the model checkpoint every logging step\n",
        "        save_steps=25,                # Save checkpoints every 50 steps\n",
        "        evaluation_strategy=\"steps\", # Evaluate the model every logging step\n",
        "        eval_steps=25,               # Evaluate and save checkpoints every 50 steps\n",
        "        do_eval=True,                # Perform evaluation at the end of training\n",
        "        report_to=\"wandb\",           # Comment this out if you don't want to use weights & baises\n",
        "        run_name=f\"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}\"          # Name of the W&B run (optional)\n",
        "    ),\n",
        "    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),\n",
        ")\n",
        "\n",
        "model.config.use_cache = False  # silence the warnings. Please re-enable for inference!\n",
        "trainer.train()"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "R9rRmDCeQiTJ"
      },
      "source": [
        "I cleared the output of the cell above because I stopped the training early, and it produced a long, ugly error message."
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "0D57XqcsyRgo"
      },
      "source": [
        "### 6. Drum Roll... Try the Trained Model!\n",
        "\n",
        "It's a good idea to kill the current process so that you don't run out of memory loading the base model again on top of the model we just trained. Go to `Kernel > Restart Kernel` or kill the process via the Terminal (`nvidia smi` > `kill [PID]`). \n",
        "\n",
        "By default, the PEFT library will only save the QLoRA adapters, so we need to first load the base model from the Huggingface Hub:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "referenced_widgets": [
            "fb8230fb86884aa6be318e2d03a88af2"
          ]
        },
        "id": "SKSnF016yRgp",
        "outputId": "bce5209d-90da-4117-c6ac-cda9f3cb3422"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "fb8230fb86884aa6be318e2d03a88af2",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/home/ubuntu/.pyenv/versions/3.10.13/lib/python3.10/site-packages/transformers/utils/hub.py:374: FutureWarning: The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.\n",
            "  warnings.warn(\n",
            "Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
        "\n",
        "base_model_id = \"mistralai/Mistral-7B-v0.1\"\n",
        "bnb_config = BitsAndBytesConfig(\n",
        "    load_in_4bit=True,\n",
        "    bnb_4bit_use_double_quant=True,\n",
        "    bnb_4bit_quant_type=\"nf4\",\n",
        "    bnb_4bit_compute_dtype=torch.bfloat16\n",
        ")\n",
        "\n",
        "base_model = AutoModelForCausalLM.from_pretrained(\n",
        "    base_model_id,  # Mistral, same as before\n",
        "    quantization_config=bnb_config,  # Same quantization config as before\n",
        "    device_map=\"auto\",\n",
        "    trust_remote_code=True,\n",
        ")\n",
        "\n",
        "eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "_BxOhAiqyRgp"
      },
      "source": [
        "Now load the QLoRA adapter from the appropriate checkpoint directory, i.e. the best performing model checkpoint:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "GwsiqhWuyRgp"
      },
      "outputs": [],
      "source": [
        "from peft import PeftModel\n",
        "\n",
        "ft_model = PeftModel.from_pretrained(base_model, \"mistral-journal-finetune/checkpoint-300\")"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "lX39ibolyRgp"
      },
      "source": [
        "and run your inference!"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "UUehsaVNyRgp"
      },
      "source": [
        "Let's try the same `eval_prompt` and thus `model_input` as above, and see if the new finetuned model performs better. I like playing with the repetition penalty (just little tweaks of .01-.05 at a time). THIS IS SO FUN. I'm obsessed wth this AI version of myself."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lMkVNEUvyRgp",
        "outputId": "7d49d409-5dbe-4306-c1a4-9d87e3073397"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            " The following is a note by Eevee the Dog, which doesn't share anything too personal: # \n",
            "I’m grateful for my best friend coming to visit me. I know we’ll have so much fun and our relationship will continue to flourish. We really are each other’s number one fan and it’s such a beautiful thing. She supports me in all that I do and celebrates my successes with joy and excitement. I am excited to show her around SF and take her to some of my favorite places. I hope she gets to meet some of my friends here as\n"
          ]
        }
      ],
      "source": [
        "eval_prompt = \" The following is a note by Eevee the Dog, which doesn't share anything too personal: # \"\n",
        "model_input = eval_tokenizer(eval_prompt, return_tensors=\"pt\").to(\"cuda\")\n",
        "\n",
        "ft_model.eval()\n",
        "with torch.no_grad():\n",
        "    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True))"
      ]
    },
    {
      "attachments": {},
      "cell_type": "markdown",
      "metadata": {
        "id": "VCJnpZoayRgq"
      },
      "source": [
        "### Sweet... it worked! The fine-tuned model now prints out journal entries in my style!\n",
        "\n",
        "How funny to see it write like me as an angsty teenager, and honestly adult. I am obsessed. It knows who my friends are and talks about them, and covers the same topics I usually cover. It's really cool.\n",
        "\n",
        "That output is quite private but I wanted you to see an example run, so I tweaked the `eval_prompt` so that it explicitly wouldn't say anything too sensitive, haha.\n",
        "\n",
        "I hope you enjoyed this tutorial on fine-tuning Mistral on your own data. If you have any questions, feel free to reach out to us on [X](https://x.com/brevdev) or [Discord](https://discord.gg/RN2a436M73).\n",
        "\n",
        "🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": []
    },
    "gpuClass": "standard",
    "kernelspec": {
      "display_name": "Python 3 (ipykernel)",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.10.12"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
