Export pretrained DETR to onnx, and inference using vortex runtime.
The exporting part use default export from pytorch, ``torch.onnx.export``,
then additional metadata is embedded to model using vortex' utility :py:mod:`~vortex.runtime.onnx.graph_ops.EmbedModelProperty`.
Vortex provides inference and visualization helper via :py:mod:`~vortex.runtime.helper.InferenceHelper`.

.. raw:: html

    <a href="https://colab.research.google.com/drive/1VO2NvjcM7HIpY0SryTc3gmr64XJg05fq">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>

.. include:: ../_build/examples/export/export.rst