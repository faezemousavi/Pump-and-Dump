# -*- coding:utf-8 -*-
import os
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tqdm import tqdm
from sklearn import metrics
import pandas as pd
import numpy as np

from data_loader import FeatGenerator
from modeling import DNN, SNN, SNNTA


flags = tf.flags
FLAGS = flags.FLAGS

# -----------------------------
# I/O
# -----------------------------
flags.DEFINE_string(
    "train_input_file",
    "/content/drive/MyDrive/entezarivoive/train_sample.csv",
    "Train input file.")

flags.DEFINE_string(
    "valid_input_file",
    "/content/drive/MyDrive/entezarivoive/valid_sample.csv",
    "Validation input file for early stopping.")

flags.DEFINE_string(
    "test_input_file",
    "TargetCoinPrediction/SeqModel/test_sample.csv",
    "Test input file (used when do_eval=True).")

flags.DEFINE_string(
    "checkpointDir",
    "TargetCoinPrediction/SeqModel/ckpt",
    "Directory to write checkpoints.")

# -----------------------------
# Hyper-parameters
# -----------------------------
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (optional).")
flags.DEFINE_integer("max_seq_length", 10, "")
flags.DEFINE_bool("do_train", True, "")
flags.DEFINE_bool("do_eval", False, "")
flags.DEFINE_integer("batch_size", 256, "")
flags.DEFINE_integer("epoch", 30, "Used by FeatGenerator repeat().")
flags.DEFINE_float("learning_rate", 5e-4, "")
flags.DEFINE_integer("num_train_steps", 1000000, "Training steps.")
flags.DEFINE_integer("num_warmup_steps", 100, "Warmup steps (if used in modeling).")
flags.DEFINE_integer("save_checkpoints_steps", 8000, "")
flags.DEFINE_float("dropout_rate", 0.2, "Dropout rate during training.")
flags.DEFINE_string("model", "dnn", "model type {dnn, snn, snnta}")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU.")
flags.DEFINE_string("data_dir", "./data/", "data dir.")

# -----------------------------
# Early stopping
# -----------------------------
flags.DEFINE_integer("early_stop_patience", 5, "Stop after N validations without improvement.")
flags.DEFINE_float("early_stop_min_delta", 1e-4, "Min AUC improvement to count as better.")
flags.DEFINE_integer("eval_every_ckpt", 1, "Validate every N saved checkpoints.")


def HitRatio_calculation(channel_ids, coin_ids, timestamps, labels, pre_probas):
    channel_ids = [x.decode("utf-8") for x in channel_ids]
    timestamps = [x.decode("utf-8") for x in timestamps]

    test_df = pd.DataFrame({"label": labels, "y_pred_proba": pre_probas}, dtype=np.float64)
    df1 = pd.DataFrame({"channel_id": channel_ids, "timestamp": timestamps}, dtype=np.int64)
    coins_df = pd.DataFrame({"coin": coin_ids}, dtype=str)
    test_df = pd.concat([test_df, df1, coins_df], axis=1)

    def hitrate(k):
        def udf(df):
            X = list(zip(df.y_pred_proba, df.label))
            X.sort(key=lambda x: x[0], reverse=True)
            top_labels = [X[i][1] for i in range(min(k, len(X)))]
            return np.array([[df.iloc[0]["channel_id"], df.iloc[0]["timestamp"], np.sum(top_labels)]])

        x_test = (
            test_df[["channel_id", "timestamp", "label", "y_pred_proba"]]
            .groupby(["channel_id", "timestamp"])
            .apply(udf)
        )

        test_hitrate = pd.DataFrame(
            np.concatenate(x_test.values, axis=0),
            columns=["channel_id", "timestamp", "label_num"],
        )
        test_hitrate.label_num = test_hitrate.label_num.astype(int)
        return test_hitrate.label_num.mean()

    HR1 = hitrate(1)
    HR3 = hitrate(3)
    HR5 = hitrate(5)
    HR10 = hitrate(10)
    HR20 = hitrate(20)
    HR50 = hitrate(50)

    return HR1, HR3, HR5, HR10, HR20, HR50


def model_fn(tensor_dict, is_training: bool):
    """Build model with explicit training/eval mode (avoid relying on FLAGS.do_train)."""
    model_config = {
        "is_training": is_training,
        "dropout_rate": FLAGS.dropout_rate if is_training else 0.0,
        "max_seq_length": FLAGS.max_seq_length,
        "learning_rate": FLAGS.learning_rate,
        "batch_size": FLAGS.batch_size,
    }

    if FLAGS.model == "snn":
        model = SNN(tensor_dict, model_config)
    elif FLAGS.model == "snnta":
        model = SNNTA(tensor_dict, model_config)
    elif FLAGS.model == "dnn":
        model = DNN(tensor_dict, model_config)
    else:
        raise ValueError("Unknown model name: %s" % FLAGS.model)

    model.build()
    return model


def eval_ckpt_auc(ckpt_path: str, input_file: str):
    """
    Evaluate a saved checkpoint on input_file and return AUC.
    This builds a fresh TF graph to avoid polluting the training graph.
    """
    tf.reset_default_graph()

    feat_generator = FeatGenerator(input_file, epoch=1, batch_size=FLAGS.batch_size)
    model = model_fn(feat_generator.tensor_dict, is_training=False)
    saver = tf.train.Saver()

    pre_probas, labels = [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_path)

        while True:
            try:
                y_, label = sess.run([model.y_hat, model.label])
                pre_probas.extend(list(y_))
                labels.extend(list(label))
            except (tf.errors.OutOfRangeError, Exception):
                break

    if not labels or not pre_probas:
        return None

    fpr, tpr, _ = metrics.roc_curve(labels, pre_probas, pos_label=1)
    return metrics.auc(fpr, tpr)


def main(_):
    os.makedirs(FLAGS.checkpointDir, exist_ok=True)

    # A "pseudo-epoch" in your old logic (batch_size * 200 samples)
    sample_num = FLAGS.batch_size * 200
    save_iter = int(sample_num / FLAGS.batch_size)  # this is 200 with your current numbers

    # -----------------------------
    # TRAINING
    # -----------------------------
    if FLAGS.do_train:
        # build TRAIN graph once
        feat_generator = FeatGenerator(FLAGS.train_input_file, FLAGS.epoch, FLAGS.batch_size)
        model = model_fn(feat_generator.tensor_dict, is_training=True)
        saver = tf.train.Saver(max_to_keep=50)

        # early stopping state
        best_auc = -1.0
        bad_count = 0
        ckpt_index = 0

        # running train stats
        pred_probas = []
        labels = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            it = 0
            start = time.time()

            for _ in tqdm(range(FLAGS.num_train_steps)):
                try:
                    _, loss, l, y_ = sess.run([model.optimizer, model.loss, model.label, model.y_hat])

                    pred_probas.extend(list(y_))
                    labels.extend(list(l))

                    if it % 50 == 0 and it > 0:
                        end = time.time()
                        fpr, tpr, _ = metrics.roc_curve(labels, pred_probas, pos_label=1)
                        auc_value = metrics.auc(fpr, tpr)
                        print("iter=%d, loss=%f, train_auc=%f, time=%.2fs" % (it, loss, auc_value, end - start))
                        pred_probas, labels = [], []
                        start = time.time()

                    it += 1

                    # Save checkpoint each "save_iter" steps (same as your old logic)
                    if it % save_iter == 0 and it > 4:
                        ckpt_index += 1
                        ckpt_path = saver.save(
                            sess,
                            os.path.join(FLAGS.checkpointDir, FLAGS.model + str(round(it / save_iter))),
                        )
                        print("Saved checkpoint:", ckpt_path)

                        # ---- EARLY STOPPING: validate every N checkpoints ----
                        if ckpt_index % FLAGS.eval_every_ckpt == 0:
                            val_auc = eval_ckpt_auc(ckpt_path, FLAGS.valid_input_file)
                            print("VALID AUC:", val_auc)

                            if val_auc is None:
                                print("Validation produced no data; skipping early-stop check.")
                            elif val_auc > best_auc + FLAGS.early_stop_min_delta:
                                best_auc = val_auc
                                bad_count = 0
                                print("✅ New best VALID AUC:", best_auc)
                            else:
                                bad_count += 1
                                print("⚠️ No improvement. bad_count=%d/%d" % (bad_count, FLAGS.early_stop_patience))

                                if bad_count >= FLAGS.early_stop_patience:
                                    print("🛑 Early stopping triggered. Best VALID AUC:", best_auc)
                                    break

                except (tf.errors.OutOfRangeError, Exception) as e:
                    print("Training stopped due to exception:", e)
                    break

    # -----------------------------
    # EVALUATION (unchanged behavior)
    # -----------------------------
    elif FLAGS.do_eval:
        feat_generator = FeatGenerator(FLAGS.test_input_file, 1, FLAGS.batch_size)
        model = model_fn(feat_generator.tensor_dict, is_training=False)

        auc_value_list = []
        num_epochs = int(FLAGS.num_train_steps / save_iter)

        for epoch in range(1, num_epochs + 1):
            ckpt = os.path.join(FLAGS.checkpointDir, FLAGS.model + str(epoch))
            saver = tf.train.Saver()

            pre_probas, labels = [], []
            coin_ids, channel_ids, timestamps = [], [], []

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                try:
                    saver.restore(sess, ckpt)
                except Exception as e:
                    print("Could not restore:", ckpt, "err:", e)
                    break

                while True:
                    try:
                        channel_id, coin_id, timestamp, y_, label, loss = sess.run(
                            [
                                feat_generator.features["channel"],
                                feat_generator.features["coin"],
                                feat_generator.features["timestamp"],
                                model.y_hat,
                                model.label,
                                model.loss,
                            ]
                        )

                        pre_probas.extend(list(y_))
                        labels.extend(list(label))
                        timestamps.extend(list(timestamp))
                        channel_ids.extend(list(channel_id))
                        coin_ids.extend(list(coin_id))

                    except (tf.errors.OutOfRangeError, Exception):
                        break

            if labels and pre_probas:
                fpr, tpr, _ = metrics.roc_curve(y_true=labels, y_score=pre_probas, pos_label=1)
                auc_value = metrics.auc(fpr, tpr)
                auc_value_list.append(auc_value)

                print("=====================================")
                print("epoch=%d, auc=%f" % (epoch, auc_value))

                HR1, HR3, HR5, HR10, HR20, HR50 = HitRatio_calculation(
                    channel_ids, coin_ids, timestamps, labels, pre_probas
                )
                print("HitRate@1=%.4f" % HR1)
                print("HitRate@3=%.4f" % HR3)
                print("HitRate@5=%.4f" % HR5)
                print("HitRate@10=%.4f" % HR10)
                print("HitRate@20=%.4f" % HR20)
                print("HitRate@50=%.4f" % HR50)

    else:
        raise ValueError("Only TRAIN (do_train) or EVAL (do_eval) mode supported.")


if __name__ == "__main__":
    tf.app.run()
