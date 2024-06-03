
for i in {2..12}
    do
        for j in {1..10}
            do
		echo "test $j"
		mpiexec -n $i -tag-output -hostfile /nfs/hpc-unime-exam/hosts -display-map python /nfs/hpc-unime-exam/hpc-exam.py
		mv /nfs/*stats.json /nfs/hpc-unime-exam/resultsdata/
            done
    done
/nfs/hpc-unime-exam/pulizia.sh
