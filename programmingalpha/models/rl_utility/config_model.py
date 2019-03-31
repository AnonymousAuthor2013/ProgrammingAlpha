"""Configurations of Transformer model
"""
import texar as tx

random_seed = 3719753
beam_width = 20
alpha = 0.6
hidden_dim = 768
max_seq_length=512
max_decoding_length = 384

emb = {
    'name': 'lookup_table',
    'dim': hidden_dim,
    'initializer': {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': hidden_dim**-0.5,
        },
    }
}

decoder1 = {
    'dim': hidden_dim,
    'num_blocks': 1,
    'multihead_attention': {
        'num_heads': 8,
        'output_dim': hidden_dim
        # See documentation for more optional hyperparameters
    },
    'position_embedder_hparams': {
        'dim': hidden_dim
    },
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'poswise_feedforward': tx.modules.default_transformer_poswise_net_hparams(
        output_dim=hidden_dim)
}

decoder = {
        # Same as in TransformerEncoder
        "scale_embeds": True,
        "num_blocks": 4,
        "dim": hidden_dim,
        'position_embedder_type': 'sinusoids',
        'position_size': max_seq_length,
        "position_embedder_hparams": {
            "dim":hidden_dim
        },
        "embedding_dropout": 0.1,
        "residual_dropout": 0.1,
        "multihead_attention": {
            'name': 'multihead_attention',
            'num_units': hidden_dim,
            'num_heads': 12,
            'dropout_rate': 0.1,
            'output_dim': hidden_dim,
            'use_bias': True,
        },
        "initializer": {
            "type": "variance_scaling_initializer",
            "kwargs": {
                "scale": 1.0,
                "mode": "fan_avg",
                "distribution": "uniform",
                },
         },

        "name": "transformer_decoder",

        # Additional for TransformerDecoder
        "embedding_tie": True,
        "output_layer_bias":True,
        "max_decoding_length": max_decoding_length,
        "poswise_feedforward":{
        "layers": [
            {
                "type": "Dense",
                "kwargs": {
                    "name": "intermediate",
                    "units": hidden_dim*4,
                    "activation": "relu",
                    "use_bias": True,
                }
            },
            {
                "type": "Dropout",
                "kwargs": {
                    "rate": 0.1,
                }
            },
            {
                "type": "Dense",
                "kwargs": {
                    "name": "out",
                    "units": hidden_dim,
                    "use_bias": True,
                }
            },


        ],
        "name": "ffn"
    }
}

loss_label_confidence = 0.9

opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.997,
            'epsilon': 1e-9
        }
    }
}

lr = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (hidden_dim ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 16000,
}

agent = {
    'discount_factor': 0.,
    'entropy_weight': .5
}
