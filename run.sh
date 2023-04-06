gcloud builds submit --config cloudbuild.yaml
gcloud ai custom-jobs create \
--region=us-central1 \
--display-name=aze-ds-firstjob \
--worker-pool-spec=machine-type=n1-standard-8,replica-count=1,container-image-uri=gcr.io/azedstesting/aze-ds-vertex
