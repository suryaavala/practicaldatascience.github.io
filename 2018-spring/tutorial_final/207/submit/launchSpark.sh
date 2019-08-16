echo "This script would launch spark&zeppelin on Google Dataproc"
gcloud dataproc clusters create sparkmlgpu \
--initialization-actions gs://dataproc-40cee9d2-1670-480a-8172-f1c4645b382a-us/zeppelin.sh
echo "The cluster has been launched"
