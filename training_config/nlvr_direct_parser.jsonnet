local train_data = std.extVar("TRAIN_DATA");
local dev_data = std.extVar("DEV_DATA");

{
  "dataset_reader": {
    "type": "nlvr",
    "lazy": false,
    "output_agendas": false,
    "mode": "train"
  },
  "validation_dataset_reader": {
    "type": "nlvr",
    "lazy": false,
    "output_agendas": false,
    "mode": "test"
  },

  "vocabulary": {
    "non_padded_namespaces": ["rule_labels", "denotations"]
  },
  "train_data_path": train_data,
  "validation_data_path": dev_data,
  "model": {
    "type": "nlvr_direct_parser",
    "sentence_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 50,
          "trainable": true
        }
      }
    },
    "action_embedding_dim": 50,
    "encoder": {
      "type": "lstm",
      "input_size": 50,
      "hidden_size": 30,
      "num_layers": 1,
      "bidirectional": true
    },
    "decoder_beam_search": {
      "beam_size": 10
    },
    "max_decoding_steps": 12,
    "attention": {"type": "dot_product"},
    "dropout": 0.2
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["sentence"],
      "batch_size": 4
    }
  },
  "trainer": {
    "checkpointer": {"num_serialized_models_to_keep": 1},
    "num_epochs": 50,
    "patience": 10,
    "cuda_device": 0,
    "validation_metric": "+denotation_accuracy",
    "optimizer": {
      "type": "adam",
      "lr": 0.005
    }
  }
}
