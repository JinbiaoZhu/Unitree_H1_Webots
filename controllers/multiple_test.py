from controller import Supervisor, PositionSensor

joint_name = "left_hip_yaw_joint"

Supervisor = Supervisor()

robot = Supervisor.getFromDef("H1")
print(robot.getPosition())

robot = Supervisor.getFromDef("H2")
ps = PositionSensor("left_hip_yaw_joint").enable(1)
res = ps.getValues()
print(robot.getField(joint_name))
print(res)

robot = Supervisor.getFromDef("H3")
print(robot.getPosition())

robot = Supervisor.getFromDef("H4")
print(robot.getPosition())

