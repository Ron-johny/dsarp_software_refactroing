# DSARP - Group1



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

* [Create](https://docs.gitlab.com/user/project/repository/web_editor/#create-a-file) or [upload](https://docs.gitlab.com/user/project/repository/web_editor/#upload-a-file) files
* [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.uni-paderborn.de/ankitah/dsarp-group1.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

* [Set up project integrations](https://git.uni-paderborn.de/ankitah/dsarp-group1/-/settings/integrations)

## Collaborate with your team

* [Invite team members and collaborators](https://docs.gitlab.com/user/project/members/)
* [Create a new merge request](https://docs.gitlab.com/user/project/merge_requests/creating_merge_requests/)
* [Automatically close issues from merge requests](https://docs.gitlab.com/user/project/issues/managing_issues/#closing-issues-automatically)
* [Enable merge request approvals](https://docs.gitlab.com/user/project/merge_requests/approvals/)
* [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

* [Get started with GitLab CI/CD](https://docs.gitlab.com/ci/quick_start/)
* [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/user/application_security/sast/)
* [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/topics/autodevops/requirements/)
* [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/user/clusters/agent/)
* [Set up protected environments](https://docs.gitlab.com/ci/environments/protected_environments/)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.


# HPC vLLM Inference & Local Integration Guide

This guide explains how to launch the **Qwen2.5-Coder-14B-Instruct** model on the **Noctua 2 HPC cluster** using **Apptainer**, and securely expose the inference endpoint to your local machine for integration with the Software Architectural Refactoring Pipeline.

---

# Part 1: Launching vLLM on the HPC Cluster

To avoid shared library issues (such as `libpython3.11.so.1.0`) that can occur with host virtual environments, we use a self-contained **Apptainer sandbox** containing its own Python environment, CUDA dependencies, and the vLLM framework.

## 0. Navigate to the Project Directory

```bash
cd /scratch/hpc-prf-dssecs/pg_arc_ref/hpc_api
```

---

## 1. Allocate GPU Resources

Log into **Noctua 2** and allocate a GPU node with **at least two A100 GPUs** for tensor parallelism.

### Option 1: Interactive `srun`

```bash
srun \
    --nodes=1 \
    --tasks-per-node=1 \
    --partition=dgx \
    --qos=devel \
    --gres=gpu:a100:2 \
    --mem=100G \
    --time=03:00:00 \
    --pty bash
```

### Option 2: `salloc`

```bash
salloc \
    -A <your_project_account> \
    -p gpu \
    --gres=gpu:a100:2 \
    --time=05:00:00
```

---

## 2. Keep the Server Running with tmux

To prevent the server from stopping if your SSH connection drops, run it inside a `tmux` session.

Create a new session:

```bash
tmux new -s vllm_hosting
```

Useful commands:

```bash
# Detach from the session
Ctrl + B, then D

# Reattach later
tmux attach -t vllm_hosting
```

---

## 3. Launch the vLLM Server

From the root of the `hpc_api` directory, start the Apptainer container.

```bash
apptainer run --nv ./vllm_sandbox \
    --model ./models/Qwen2.5-Coder-14B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 16384 \
    --tensor-parallel-size 2
```

> **Important:** Note the hostname of the compute node assigned by Slurm (for example, `n2dgx01`). You will need this hostname when creating the SSH tunnel.

---

# Part 2: Accessing the Model from Your Local Machine

The Noctua 2 compute nodes are not publicly accessible. To access the running vLLM server from your local computer, create an **SSH port-forwarding tunnel**.

## 1. Create the SSH Tunnel

On your **local machine**, open a new terminal (or Git Bash), navigate to the directory containing your SSH private key (`id_ed25519`), and execute:

```bash
ssh \
    -i ./id_ed25519 \
    -o ProxyCommand="ssh -i ./id_ed25519 -W %h:%p ankitah@fe.noctua2.pc2.uni-paderborn.de" \
    -L 8000:n2dgx01:8000 \
    ankitah@n2login3
```

> **Note:** Replace `n2dgx01` with the hostname of the compute node allocated in the previous step if it differs.

Keep this terminal window open for the duration of your session.

Once connected, your local machine can access the vLLM server at:

```
http://localhost:8000
```

---

# Part 3: Using the Model in Python

With the SSH tunnel active, your local machine can communicate with the remote model using the OpenAI-compatible vLLM API.

## 1. Install the OpenAI Python Client

```bash
pip install openai
```

---

## 2. Example Integration

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="hpc-token",  # Dummy value; authentication is not enforced by vLLM
)

def analyze_code_architecture(snippet: str) -> str:
    response = client.chat.completions.create(
        model="./models/Qwen2.5-Coder-14B-Instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert software architect specializing "
                    "in software refactoring and modular design."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Refactor the following code for improved performance "
                    "and modularity:\n\n"
                    f"{snippet}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    code = "def f(x): return [i for i in x if x.count(i) == 1]"
    print(analyze_code_architecture(code))
```

---

# Workflow Summary

1. Navigate to the `hpc_api` directory.
2. Allocate a GPU node with two A100 GPUs.
3. Start a `tmux` session.
4. Launch the vLLM server inside the Apptainer sandbox.
5. Record the allocated compute node hostname.
6. Create an SSH port-forwarding tunnel from your local machine.
7. Connect to the model using the OpenAI Python client via `http://localhost:8000/v1`.