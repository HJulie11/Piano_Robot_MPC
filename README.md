# Piano-Playing Robot: Task and Motion Planning

To install submodules before run simulation run the command below:
```sh
git clone https://github.com/HJulie11/Piano_Robot_MPC.git
cd robopianist/
rm -rf third_party/mujoco_menagerie
git submodule add -f https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git submodule init && git submodule update
bash scripts/install_deps.sh
conda create -n pianist python=3.10
conda activate pianist
pip install -e ".[dev]"
```

To run the simulation:
```sh
python examples/piano_with_one_shadow_hand_env.py --headless
```

Inverse Kinematics Demo
https://github.com/HJulie11/Piano_Robot_MPC/blob/main/ik%20(manual)%20press.mov

Motion Planning Only Demo
https://github.com/HJulie11/Piano_Robot_MPC/blob/main/Only%20MP.mov

MP + Dynamics Demo
https://github.com/HJulie11/Piano_Robot_MPC/blob/main/MPnDynamics.mov

MP + Dynamics + Controller Demo
https://github.com/HJulie11/Piano_Robot_MPC/blob/main/controll.mov
