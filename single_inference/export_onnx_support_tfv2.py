import sys

sys.path.insert(0, ".")
import tensorflow as tfv2
tf = tfv2.compat.v1
import os
import shutil
import argparse
from typing import List
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.saved_model import tag_constants
from tensorflow.python import ops
import network
import subprocess
import onnx
from onnxsim import simplify

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser(description="Freeze your network")
parser.add_argument("--ckpt", type=str, help="which checkpoint freeze?", default="ckpt/pydnet")
parser.add_argument("--arch", type=str, help="network to freeze", default="pydnet")
parser.add_argument("--dest", type=str, help="where to save frozen models", default=".")
parser.add_argument("--height", type=int, default=384, help="height of image")
parser.add_argument("--width", type=int, default=640, help="width of image")
parser.add_argument("--opset", type=int, default=13, help="onnx opset")
args = parser.parse_args()


def main(_):
    params = {
        "arch": args.arch,
        "output": os.path.join(args.dest, "frozen_models"),
        "protobuf": "frozen_" + args.arch + ".pb",
        "pbtxt": args.arch + ".pbtxt",
        "ckpt": args.arch + ".ckpt",
        "mlmodel": args.arch + ".mlmodel",
        "onnx": args.arch + ".onnx",
        "input_saver_def_path": "",
        "input_binary": False,
        "restore_op": "save/restore_all",
        "saving_op": "save/Const:0",
        "frozen_graph_name": "frozen_" + args.arch + ".pb",
        "optimized_graph_name": "optimized_" + args.arch + ".pb",
        "optimized_tflite_name": "tflite_" + args.arch + ".tflite",
        "clear_devices": True,
    }

    if not os.path.exists(params["output"]):
        os.makedirs(params["output"])

    with tf.Graph().as_default():

        network_params = {
            "height": args.height,
            "width": args.width,
            "is_training": False,
        }
        input_node = "im0"
        input_tensor = tf.placeholder(
            tf.float32,
            [1, network_params["height"], network_params["width"], 3],
            name="im0",
        )
        model = network.Pydnet(network_params)
        predictions = model.forward(input_tensor)
        params["output_nodes_port"] = [x.name for x in model.output_nodes]
        params["output_nodes"] = [
            out.name.replace(":0", "") for out in model.output_nodes
        ]
        print("=> output nodes port: {}".format(params["output_nodes_port"]))
        print("=> output nodes: {}".format(params["output_nodes"]))
        params["input_nodes"] = [input_node]
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, args.ckpt)
            tf.train.write_graph(sess.graph_def, params["output"], params["pbtxt"])
            graph_pbtxt = os.path.join(params["output"], params["pbtxt"])
            graph_path = os.path.join(params["output"], params["ckpt"])
            saver.save(sess, graph_path)

            outputs = params["output_nodes"][0]
            for name in params["output_nodes"][1:]:
                outputs += "," + name

            frozen_graph_path = os.path.join(
                params["output"], params["frozen_graph_name"]
            )

            graph = tf.get_default_graph()
            sess = tf.Session()
            saver = tf.train.import_meta_graph(f'{params["output"]}/{args.arch}.ckpt.meta')
            saver.restore(sess, f'{params["output"]}/{args.arch}.ckpt')
            tf.train.write_graph(sess.graph_def, f'{params["output"]}', f'{args.arch}.pb', as_text=False)

            def get_graph_def_from_file(graph_filepath: str):
                with tf.Graph().as_default():
                    with tf.gfile.GFile(graph_filepath, 'rb') as f:
                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(f.read())
                        return graph_def

            def convert_graph_def_to_saved_model(export_dir: str, graph_filepath: str, input_name: str, outputs: List[str]):
                graph_def = get_graph_def_from_file(graph_filepath)
                with tf.compat.v1.Session(graph=tf.Graph()) as session:
                    tf.import_graph_def(graph_def, name='')
                    tf.compat.v1.saved_model.simple_save(
                        session,
                        export_dir,
                        inputs={input_name: session.graph.get_tensor_by_name('{}:0'.format(node.name))
                            for node in graph_def.node if node.op=='Placeholder'},
                        outputs={t.rstrip(":0"):session.graph.get_tensor_by_name(t) for t in outputs}
                    )
                    print('Graph converted to SavedModel!')

        onnx_file_path = f"pydnet_{args.height}x{args.width}.onnx"
        command = [
            "python", "-m", "tf2onnx.convert",
            "--checkpoint", f'{params["output"]}/{args.arch}.ckpt.meta',
            "--output", onnx_file_path,
            "--inputs", "im0:0",
            "--outputs", "truediv:0",
            "--inputs-as-nchw", "im0:0",
            "--opset", str(args.opset),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("Success:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error:", e.stderr)

        model = onnx.load(onnx_file_path)
        model_simp, check = simplify(model)
        onnx.save(model_simp, onnx_file_path)

    print("Done!")


if __name__ == "__main__":
    tf.app.run()
