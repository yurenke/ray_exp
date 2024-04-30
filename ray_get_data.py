# from typing import Dict
import tempfile
import os
import json
import numpy as np
import psycopg2
from psycopg2 import Error
import logging
from opencc import OpenCC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import transformers
from transformers import DistilBertTokenizer

import ray
from ray import train
from ray.train import ScalingConfig, Checkpoint
from ray.train.tensorflow import TensorflowTrainer
from ray.train import RunConfig, CheckpointConfig

DATABASE_ENGINE = "django.db.backends.postgresql_psycopg2"
DATABASE_NAME = "ai_chat_filter"
DATABASE_USER = "postgres"
DATABASE_PASSWORD = "postgres"
DATABASE_HOST = "localhost"
DATABASE_PORT = 5432
TABLE_NAME = "ai_textbooksentense"
SEN_MAXLEN = 20
NUM_CLASSES = 3

tokenizer = DistilBertTokenizer.from_pretrained('./tokenizer', local_files_only=True)


def trim_only_general_and_chinese(string):
    _result = ''

    for uc in string:
        _code = ord(uc)

        if _code >= 0x4e00 and _code <= 0x9faf:
            # chinese
            _result += uc
            continue
        elif _code == 0x3000 or _code == 0x00a0:
            # full space to half space
            _code = 0x0020
        elif _code > 0xfee0 and _code < 0xffff:
            # full char to half
            _code -= 0xfee0

        if _code == 0x0020 or (_code >= 0x0030 and _code <= 0x0039) or (_code >= 0x0041 and _code <= 0x005a) or (_code >= 0x0061 and _code <= 0x007a):
            _result += chr(_code).lower()
            
    return _result

def get_data():
    print("getting data from DB....")
    df = None

    try:
        # Connect to an existing database
        connection = psycopg2.connect(database=DATABASE_NAME, 
                user=DATABASE_USER, 
                password=DATABASE_PASSWORD, 
                host=DATABASE_HOST, 
                port=DATABASE_PORT)

        # Create a cursor to perform database operations
        cur = connection.cursor()
        # Executing a SQL query
        ## sql query
        sql = f"""
                SELECT id,
                    origin,
                    text,
                    status
                FROM {TABLE_NAME}
                ORDER BY id DESC
                LIMIT 800000
                """
                
                ## 執行sql語法
        cur.execute(sql)
        # Fetch result
        name = [desc[0] for desc in cur.description]
                
        ## 取得資料
        df = pd.DataFrame(cur.fetchall(),columns=name)

        # logging.info(f"total records: {len(df.index)}")
        print(f"total records: {len(df.index)}")

    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if (connection):
            cur.close()
            connection.close()
            print("PostgreSQL connection is closed")

    return df

def transform_data(df):
    def _transform_status(x):
        return x if x < 2 else 2
    cc = OpenCC('t2s')

    df = df[df.status != 3]
    df['text'] = df['text'].apply(lambda x: cc.convert(x))
    df['text'] = df['text'].apply(lambda x: trim_only_general_and_chinese(x).strip())
    df = df[df['text'].astype(bool)] # remove rows containing empty string
    df = df.drop_duplicates(subset=['text'])
            
    df['target'] = df['status'].apply(lambda x: _transform_status(x))
    df = df[['text', 'target']]
    # print(df.dtypes)
    return df

def build_model():
    max_len = SEN_MAXLEN
    transformer_layer = (
        transformers.TFDistilBertModel
        .from_pretrained("distilbert-base-multilingual-cased")
    )
    
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer_layer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(NUM_CLASSES, activation='softmax')(cls_token)
        
    model = Model(inputs=input_word_ids, outputs=out)
    return model
    # self.model.compile(Adam(learning_rate=self.parameters['lr']), loss='sparse_categorical_crossentropy', weighted_metrics=['accuracy'])
        
    # self.model.summary() 

def train_func(config: dict):
    batch_size = config.get("batch_size", 64) # * train.get_context().get_world_size()
    epochs = config.get("epochs", 3)
    lr = config.get('lr', 1e-5)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = build_model()
        multi_worker_model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            weighted_metrics=['accuracy'],
        )

    train_dataset = train.get_dataset_shard("train")
    val_dataset = train.get_dataset_shard("val")

    results = []
    for epoch in range(epochs):
        train_tf_dataset = train_dataset.to_tf(
            feature_columns="input_ids", label_columns="labels", batch_size=batch_size
        )
        val_tf_dataset = val_dataset.to_tf(
            feature_columns="input_ids", label_columns="labels", batch_size=batch_size
        )
        history = multi_worker_model.fit(
            x=train_tf_dataset,
            class_weight=config.get("cls_weights"),
            validation_data=val_tf_dataset,
            # callbacks=[ReportCheckpointCallback()]
        )

        with tempfile.TemporaryDirectory(dir="/home/stanley/ray_exp/checkpoints") as temp_checkpoint_dir:
            multi_worker_model.save(os.path.join(temp_checkpoint_dir, "model.keras"))
            checkpoint_dict = os.path.join(temp_checkpoint_dir, "checkpoint.json")
            with open(checkpoint_dict, "w") as f:
                json.dump({"epoch": epoch}, f)
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            train.report({"loss": history.history["loss"][0]}, checkpoint=checkpoint)
        # results.append(history.history)
    # return results


def tokenize_function(batch):
    outputs = tokenizer(
            list(batch['text']),
            truncation=True,
            padding="max_length",
            return_tensors="tf",
            return_attention_mask=False,
            return_token_type_ids=False,
            max_length=SEN_MAXLEN
        )
    
    outputs["labels"] = tf.convert_to_tensor(batch['target'])
    return outputs

# def tokenize_function(examples):
    # enc_di = tokenizer.batch_encode_plus(batch["text"].values.tolist(), padding="max_length", truncation=True, return_attention_mask=False, 
    #         return_token_type_ids=False,
    #         max_length=SEN_MAXLEN)
    # batch["text"] = np.asarray(enc_di['input_ids']).astype('int32')
    
    # examples['text'] = np.asarray(tokenizer(examples["text"], padding="max_length", truncation=True, return_attention_mask=False, 
    #         return_token_type_ids=False,
    #         max_length=SEN_MAXLEN)['input_ids']).astype('int32')
    # return examples

def main():
    df = get_data()
    df = transform_data(df)
    print("total rows after dedup: ", len(df.index))
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['target'])
    # ds = ray.data.from_pandas(df)
    # train, val = ds.train_test_split(test_size=0.2, shuffle=True)
    print("type of train after splitting: ", type(train))
    print("train: ", len(train_df.index))
    print("val: ", len(val_df.index))

    class_weights = [(i, _) for (i, _) in enumerate(class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(train_df.target),
                                                 y=train_df.target))]
    class_weight_dict = dict(class_weights)
    print('class weights: {}'.format(class_weight_dict))

    train_ds = ray.data.from_pandas(train_df)
    val_ds = ray.data.from_pandas(val_df)

    del train_df
    del val_df
    # train_ds = train_ds.map(tokenize_function)
    # val_ds = val_ds.map(tokenize_function)
    train_ds = train_ds.map_batches(tokenize_function, batch_size="default")
    val_ds = val_ds.map_batches(tokenize_function, batch_size="default")

    # tf_dataset = val_ds.to_tf(
    #     feature_columns="input_ids",
    #     label_columns="labels",
    #     batch_size=2
    # )
    # for features, labels in tf_dataset:
    #     print(features, labels)

    print('starting to train')
    config = {"lr": 1e-5, "batch_size": 64, "epochs": 3, "cls_weights": class_weight_dict}

    scaling_config = ScalingConfig(num_workers=2, use_gpu=False)
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            # *Best* checkpoints are determined by these params:
            checkpoint_score_attribute="val_accuracy",
            checkpoint_score_order="max",
        ),
        # This will store checkpoints on S3.
        storage_path="/home/stanley/ray_exp/best",
    )


    trainer = TensorflowTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_ds, "val": val_ds},
    )
    result = trainer.fit()
    # print(result.metrics) 
    # print(result.checkpoint)
    # Print available checkpoints
    for checkpoint, metrics in result.best_checkpoints:
        print("Loss", metrics["loss"], "checkpoint", checkpoint)

    # val_ds = val_ds.map(tokenize_function, batched=True)

    # train_ds.write_json("local:///home/stanley/ray_exp/train_data")
    # val.write_csv("local:///home/stanley/ray_exp/val_data")
    # groupby_ds = train.groupby("target").count()
    # groupby_counts = groupby_ds.take_all()
    # print("check groupby_counts type: ", type(groupby_counts))
    # print("its content", groupby_counts)

if __name__ == '__main__':
    main()