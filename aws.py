import os
import json
import boto3
from botocore.exceptions import ClientError


with open('aws_params', 'r') as f:
    Params = json.load(f)


def create_session():
    session = boto3.session.Session()
    if len(session.available_profiles) == 0:
        print('No credentials found.')
        region = input('\tWhat is your region?  ')
        id = input('\tWhat is your access key id?  ')
        key = input('\tWhat is your access key?  ')

        with open(os.path.expanduser('~/.aws/config'), 'w') as f:
            f.write('[default]\n')
            f.write(f'region={region}\n')
            f.write(f'aws_access_key_id={id}\n')
            f.write(f'aws_secret_access_key={key}')
        session = boto3.session.Session()
    return session


sess = create_session()


def get_key(key_name):
    """Create and save (or load) a key pair."""
    file_name = f'{key_name}.pem'
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            key_pair = f.read()
    else:
        ec2 = sess.resource('ec2')
        key_pair = ec2.create_key_pair(KeyName=key_name).key_material
        with open(file_name, 'w') as f:
            f.write(key_pair)
    return key_pair


def create_bucket(bucket_name, region):
    try:
        s3_client = sess.client('s3')
        location = {'LocationConstraint': region}
        s3_client.create_bucket(Bucket=bucket_name,
                                CreateBucketConfiguration=location)
    except ClientError as e:
        return False
    return True


def launch_instances(startup_script):
    """Launch a new s3 instance.

    startup_script (str): A script to execute on startup or a path containing that script.
    """
    try:
        with open(startup_script, 'r') as f:
            startup = f.read()
    except FileNotFoundError:
        pass
    ec2 = sess.resource('ec2')
    try:
        profile = Params['IamInstanceProfile']
    except KeyError:
        MissingParameter('The IamInstanceProfile parameter is missing. Please add to the'
                         'aws_params file.')

    ec2.create_instances(
        ImageId='ami-05c06f2ea67b59196',
        InstanceType='t2.micro',
        KeyName='newkey',
        MinCount=1,
        MaxCount=1,
        SecurityGroups=['game-player'],
        IamInstanceProfile={'Arn': profile},
        UserData=startup
    )


def upload_file(filename, bucket_name):
    # Create an S3 client
    s3 = sess.client('s3')

    # Uploads the given file using a managed uploader, which will split up large
    # files automatically and upload parts in parallel.
    s3.upload_file(filename, bucket_name, filename)


class MissingParameter(Exception):
    """Raised when there is a necessary parameter missing and it is needed for interacting
    with AWS. These parameters belong in the aws_params file."""