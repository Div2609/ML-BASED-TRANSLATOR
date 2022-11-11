{
 "metadata": {
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
   "version": "3.8.3"
  },
  "orig_nbformat": 2,
  "kernelspec": {
   "name": "python383jvsc74a57bd010588537bc3f81a35ceb684741d6beb7f73e874ad223cf30a51d7cc1aad9f930",
   "display_name": "Python 3.8.3 64-bit ('base': conda)"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2,
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "['  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current',\n",
       " '                                 Dload  Upload   Total   Spent    Left  Speed',\n",
       " '',\n",
       " '  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0',\n",
       " '  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0',\n",
       " '  1 6134k    1   99k    0     0  85371      0  0:01:13  0:00:01  0:01:12 85300',\n",
       " '  8 6134k    8  504k    0     0   230k      0  0:00:26  0:00:02  0:00:24  230k',\n",
       " ' 12 6134k   12  795k    0     0   248k      0  0:00:24  0:00:03  0:00:21  248k',\n",
       " ' 25 6134k   25 1541k    0     0   355k      0  0:00:17  0:00:04  0:00:13  355k',\n",
       " ' 46 6134k   46 2847k    0     0   510k      0  0:00:12  0:00:05  0:00:07  531k',\n",
       " ' 58 6134k   58 3615k    0     0   555k      0  0:00:11  0:00:06  0:00:05  660k',\n",
       " ' 80 6134k   80 4955k    0     0   686k      0  0:00:08  0:00:07  0:00:01  883k',\n",
       " ' 94 6134k   94 5774k    0     0   666k      0  0:00:09  0:00:08  0:00:01  910k',\n",
       " '100 6134k  100 6134k    0     0   691k      0  0:00:08  0:00:08 --:--:-- 1010k']"
      ]
     },
     "metadata": {},
     "execution_count": 55
    }
   ],
   "source": [
    "!!curl -O http://www.manythings.org/anki/fra-eng.zip\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_path = \"fra.txt\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 64 \n",
    "epochs = 100 \n",
    "latent_dim = 256 \n",
    "num_samples = 10000  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Input and data processing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Number of samples: 10000\nNumber of unique input tokens: 71\nNumber of unique output tokens: 93\nMax sequence length for inputs: 15\nMax sequence length for outputs: 59\n"
     ]
    }
   ],
   "source": [
    "input_texts = []\n",
    "target_texts = []\n",
    "input_characters = set()\n",
    "target_characters = set()\n",
    "with open(data_path, \"r\", encoding=\"utf-8\") as f:\n",
    "    lines = f.read().split(\"\\n\")\n",
    "for line in lines[: min(num_samples, len(lines) - 1)]:\n",
    "    input_text, target_text, _ = line.split(\"\\t\")\n",
    "    # We use \"tab\" as the \"start sequence\" character\n",
    "    # for the targets, and \"\\n\" as \"end sequence\" character.\n",
    "    target_text = \"\\t\" + target_text + \"\\n\"\n",
    "    input_texts.append(input_text)\n",
    "    target_texts.append(target_text)\n",
    "    for char in input_text:\n",
    "        if char not in input_characters:\n",
    "            input_characters.add(char)\n",
    "    for char in target_text:\n",
    "        if char not in target_characters:\n",
    "            target_characters.add(char)\n",
    "\n",
    "input_characters = sorted(list(input_characters))\n",
    "target_characters = sorted(list(target_characters))\n",
    "num_encoder_tokens = len(input_characters)\n",
    "num_decoder_tokens = len(target_characters)\n",
    "max_encoder_seq_length = max([len(txt) for txt in input_texts])\n",
    "max_decoder_seq_length = max([len(txt) for txt in target_texts])\n",
    "\n",
    "print(\"Number of samples:\", len(input_texts))\n",
    "print(\"Number of unique input tokens:\", num_encoder_tokens)\n",
    "print(\"Number of unique output tokens:\", num_decoder_tokens)\n",
    "print(\"Max sequence length for inputs:\", max_encoder_seq_length)\n",
    "print(\"Max sequence length for outputs:\", max_decoder_seq_length)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Give indexs to charcters\n",
    "input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])\n",
    "target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "##INtialise the data #D encodr_input,decoder_input and decoder_target\n",
    "encoder_input_data = np.zeros(\n",
    "    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype=\"float32\"\n",
    ")\n",
    "decoder_input_data = np.zeros(\n",
    "    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype=\"float32\"\n",
    ")\n",
    "decoder_target_data = np.zeros(\n",
    "    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype=\"float32\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "(10000, 15, 71)\n"
     ]
    }
   ],
   "source": [
    "print(encoder_input_data.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):\n",
    "    for t, char in enumerate(input_text):\n",
    "        encoder_input_data[i, t, input_token_index[char]] = 1.0\n",
    "    encoder_input_data[i, t + 1 :, input_token_index[\" \"]] = 1.0\n",
    "    for t, char in enumerate(target_text):\n",
    "        # decoder_target_data is ahead of decoder_input_data by one timestep\n",
    "        decoder_input_data[i, t, target_token_index[char]] = 1.0\n",
    "        if t > 0:\n",
    "            # decoder_target_data will be ahead by one timestep\n",
    "            # and will not include the start character.\n",
    "            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0\n",
    "    decoder_input_data[i, t + 1 :, target_token_index[\" \"]] = 1.0\n",
    "    decoder_target_data[i, t:, target_token_index[\" \"]] = 1.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define an input sequence and process it.\n",
    "encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))\n",
    "encoder = keras.layers.LSTM(latent_dim, return_state=True)\n",
    "encoder_outputs, state_h, state_c = encoder(encoder_inputs)\n",
    "\n",
    "# We discard `encoder_outputs` and only keep the states.\n",
    "encoder_states = [state_h, state_c]\n",
    "\n",
    "# Set up the decoder, using `encoder_states` as initial state.\n",
    "decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))\n",
    "\n",
    "# We set up our decoder to return full output sequences,\n",
    "# and to return internal states as well. We don't use the\n",
    "# return states in the training model, but we will use them in inference.\n",
    "decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)\n",
    "decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)\n",
    "decoder_dense = keras.layers.Dense(num_decoder_tokens, activation=\"softmax\")\n",
    "decoder_outputs = decoder_dense(decoder_outputs)\n",
    "\n",
    "# Define the model that will turn\n",
    "# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`\n",
    "model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(\n",
    "    optimizer=\"rmsprop\", loss=\"categorical_crossentropy\", metrics=[\"accuracy\"]\n",
    ")\n",
    "model.fit(\n",
    "    [encoder_input_data, decoder_input_data],\n",
    "    decoder_target_data,\n",
    "    batch_size=batch_size,\n",
    "    epochs=epochs,\n",
    "    validation_split=0.2,\n",
    ")\n",
    "# Save model\n",
    "model.save(\"french_to_eng\")"
   ]
  }
 ]
}
