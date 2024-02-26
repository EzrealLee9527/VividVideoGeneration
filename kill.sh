ps -def | grep train_a100_1.sh | cut -c 9-16 | xargs kill -9
ps -def | grep train_a100_2.sh | cut -c 9-16 | xargs kill -9
ps -def | grep train_a100_3.sh | cut -c 9-16 | xargs kill -9
ps -def | grep ysr_magicanimate | cut -c 9-16 | xargs kill -9

# ps -def | grep svd_diffusers | cut -c 9-16 | xargs kill -9
# ps -def | grep train | cut -c 9-16 | xargs kill -9