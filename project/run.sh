for m in 10 15 20 24
do 
    python exp.py --model=cnn \
        --dataset=cifar \
        --pattern=iid \
        --m_user=$m

    python exp.py --model=mlp \
        --dataset=mnist \
        --pattern=noniid \
        --dirichlet=0.5 \
        --m_user=$m

    python exp.py --model=cnn \
        --dataset=cifar \
        --pattern=noniid \
        --dirichlet=0.5 \
        --m_user=$m 

    python exp.py --model=resnet \
        --dataset=cifar \
        --pattern=noniid \
        --dirichlet=0.5 \
        --m_user=$m 
done