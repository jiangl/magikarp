# WIP

# Overview
{Coming Soon!}

# Technologies
## to explicitly install
python3.7 (https://www.python.org/downloads/)
npm (https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
rabbitmq (https://www.rabbitmq.com/download.html)
postgres (http://postgresguide.com/setup/install.html)

## to note
PyTorch
Pandas
Numpy
Flask
Celery

# Setup
`initdb -D core_insure/data/psql`
`cd core_insure && pip3 install -r requirements.txt && cd ..`
`cd ui && npm install && cd ..`

# Run Main Service
Start DB
`pg_ctl -D core_insure/data/psql -l logfile start`
Start UI
`cd ui && npm start && cd ..`
Start Server
`cd core_insure && python3 run_server.py && cd ..`
Start queueing service & workers
`rabbitmq-server`
`celery -A server worker --loglevel=info`

# Analytics for models
`jupyter notebook` --navigate to--> `model_analytics.ipynb`
