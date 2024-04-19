## 介绍

`通义千问-14B（Qwen-14B）`是阿里云研发的通义千问大模型系列的140亿参数规模的模型。Qwen-14B是基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。同时，在Qwen-14B的基础上，我们使用对齐机制打造了基于大语言模型的AI助手Qwen-14B-Chat。本仓库为Qwen-14B-Chat的仓库。

## 要求

- python 3.8及以上版本
- pytorch 1.12及以上版本，推荐2.0及以上版本
- 建议使用CUDA 11.4及以上（GPU用户、flash-attention用户等需考虑此选项）
- python 3.8 and above
- pytorch 1.12 and above, 2.0 and above are recommended
- CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)

## 依赖项

运行Qwen-14B-Chat，请确保满足上述要求，再执行以下pip命令安装依赖库

```shell
pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
```

另外，推荐安装flash-attention库（当前已支持flash attention 2），以实现更高的效率和更低的显存占用。

```shell
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .

pip install csrc/layer_norm
pip install csrc/rotary
```

## 快速使用

以下是基模：

```shell
cat <<EOF > qwen.py
from modelscope import AutoTokenizer, AutoModelForCausalLM

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-14B-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
            "qwen/Qwen-14B-Chat",
                device_map="auto",
                    trust_remote_code=True
                    ).eval()
response, history = model.chat(tokenizer, "翻译:The result is that the graduates cannot enter the professions for which they are trained and must take temporary jobs,which do not require a college degree.", history=None)
print(response)
# 结果是毕业生无法进入他们接受培训的职业，只能从事不需要大学学位的临时工作。
EOF

# 运行模型
nohup python qwen.py &

# 查看日志
tail -f nohup.out
```

## Lora微调模型

```shell
# 下载官方微调项目示例
# 将本文档下 `dataset/fy.json` 微调数据复制到 `Qwen/finetune` 下
git clone https://github.com/QwenLM/Qwen.git
cd Qwen

# 配置模型和数据路径，修改微调脚本参数
vim finetune/finetune_lora_single_gpu.sh

MODEL="/root/.cache/modelscope/hub/qwen/Qwen-14B-Chat"
DATA="/workdir/Qwen/finetune/fy.json"

# 训练完成后lora模型位于output_qwen下
bash finetune/finetune_lora_single_gpu.sh

# 合并基础模型和lora模型
cat <<EOF > merge.py
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
            "/workdir/Qwen/output_qwen", # path to the output directory
                device_map="auto",
                    trust_remote_code=True
                    ).eval()

merged_model = model.merge_and_unload()
# max_shard_size and safe serialization are not necessary. 
# They respectively work for sharding checkpoint and save the model to safetensors
merged_model.save_pretrained("/workdir/new_model", max_shard_size="2048MB", safe_serialization=True)
EOF

# 执行合并
python merge.py

# 复制必要文件到合并后的模型
cp /root/.cache/modelscope/hub/qwen/Qwen-14B-Chat/*.py /workdir/new_model/
cp /root/.cache/modelscope/hub/qwen/Qwen-14B-Chat/*.cu /workdir/new_model/
cp /root/.cache/modelscope/hub/qwen/Qwen-14B-Chat/*.cpp /workdir/new_model/
cp /root/.cache/modelscope/hub/qwen/Qwen-14B-Chat/qwen.tiktoken /workdir/new_model/
cp /root/.cache/modelscope/hub/qwen/Qwen-14B-Chat/tokeniz* /workdir/new_model/


# 测试新模型，修改模型路径
cp qwen.py qwen_new_model.py
vim qwen_new_model.py

tokenizer = AutoTokenizer.from_pretrained("/workdir/new_model", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
            "/workdir/new_model",
                device_map="auto",
                    trust_remote_code=True
                    ).eval()

python qwen_new_model.py
```

# 结果对比

```shell
# 没有微调
python qwen.py
# 结果是毕业生无法进入他们接受培训的职业，只能从事不需要大学学位的临时工作。


# 没有微调
python qwen_new_model.py
# 其结果是这些毕业生找不到与他们大学专业对口的工作，只得找一些临时的，根本用不着大学学历的工作。

```