# Build from base image
FROM funsearch-base:latest
# trick to have the commands in the container history
RUN echo "funsearch run examples_ng/dm_control_fish_upright_spec.py 5 --sandbox_type ExternalProcessSandbox" >> /root/.bash_history
RUN echo "funsearch run examples_ng/dm_control_fish_swim_spec.py 5 --sandbox_type ExternalProcessSandbox" >> /root/.bash_history

RUN echo "funsearch run examples_ng/dm_control_cheetah_spec.py 5 --sandbox_type ExternalProcessSandbox" >> /root/.bash_history

RUN echo "funsearch run examples_ng/dm_control_reacher_easy_spec.py 5 --sandbox_type ExternalProcessSandbox" >> /root/.bash_history
RUN echo "funsearch run examples_ng/dm_control_reacher_hard_spec.py 5 --sandbox_type ExternalProcessSandbox" >> /root/.bash_history

RUN echo "funsearch run examples_ng/dm_control_finger_easy_spec.py 5 --sandbox_type ExternalProcessSandbox" >> /root/.bash_history
RUN echo "funsearch run examples_ng/dm_control_finger_hard_spec.py 5 --sandbox_type ExternalProcessSandbox" >> /root/.bash_history

RUN echo "funsearch run examples_ng/dm_control_hopper_hop_spec.py 5 --sandbox_type ExternalProcessSandbox" >> /root/.bash_history
RUN echo "funsearch run examples_ng/dm_control_hopper_stand_spec.py 5 --sandbox_type ExternalProcessSandbox" >> /root/.bash_history

RUN echo "funsearch run examples_ng/dm_control_swingup_spec.py 1 --sandbox_type ExternalProcessSandbox" >> /root/.bash_history
RUN echo "funsearch run examples_ng/dm_control_ballcup_spec.py 1 --sandbox_type ExternalProcessSandbox" >> /root/.bash_history
RUN echo "funsearch run examples_ng/dm_control_double_swingup_spec.py 1 --sandbox_type ExternalProcessSandbox" >> /root/.bash_history

RUN pip uninstall  --yes  transformers
RUN pip install transformers

RUN apt update && apt install -y nano
COPY examples_ng ./examples_ng
# COPY backups ./backups

COPY funsearch ./funsearch
RUN pip install --no-deps -e  .
CMD /bin/bash && export MUJOCO_GL=osmesa
