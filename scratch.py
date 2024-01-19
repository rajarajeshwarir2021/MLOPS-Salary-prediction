# from zenml.client import Client
#
# runs = Client().list_pipeline_runs()
# run_id = runs[0].artifact_versions.pop().id
# print(run_id)
#
# artifact = Client().get_artifact_version(run_id)
# loaded_artifact = artifact.load()
# print(loaded_artifact)

# Get the name of the current pipeline run
#current_run_name = get_step_context().pipeline_run.name

# Fetch the current pipeline run
#current_run = Client().get_pipeline_run(current_run_name)

# Fetch the previous run of the same pipeline
#previous_run = current_run.pipeline.runs[1]

#run_metadata = run.run_metadata
#artifact = Client().get_artifact_version('7f2fd697-9a46-4665-9cf1-36a94ede328b')
#loaded_artifact = artifact.load()
#print(loaded_artifact)
#'7f2fd697-9a46-4665-9cf1-36a94ede328b'
#[ArtifactVersionResponse(id=UUID('8125b130-1299-4516-954c-fa7916b03358'), permission_denied=False, body=ArtifactVersionResponseBody(created=datetime.datetime(2024, 1, 18, 15, 45, 47, 943198), updated=datetime.datetime(2024, 1, 18, 15, 45, 47, 943198), user=UserResponse(id=UUID('e635e34b-d62d-4921-a256-79a5db59272b'), permission_denied=False, body=UserResponseBody(created=datetime.datetime(2024, 1, 4, 13, 34, 12, 602287), updated=datetime.datetime(2024, 1, 4, 13, 34, 12, 602287), active=True, activation_token=None, full_name='', email_opted_in=None, is_service_account=False), metadata=None, name='default'), artifact=ArtifactResponse(id=UUID('e57b4827-9b7d-43b2-a2f1-4bca77b6fdd6'), permission_denied=False, body=ArtifactResponseBody(created=datetime.datetime(2024, 1, 11, 19, 57, 22, 736592), updated=datetime.datetime(2024, 1, 11, 19, 57, 22, 736592), tags=[]), metadata=None, name='train_pipeline::read_config::output'), version='6', uri='C:\\Users\\rajar\\AppData\\Roaming\\zenml\\local_stores\\dbc029b4-f329-4ec0-88bd-c107eba71b9e\\read_config\\output\\6b12355f-2235-4af3-b462-9f6db1f83a3b', type=<ArtifactType.DATA: 'DataArtifact'>, materializer=Source(module='zenml.materializers.cloudpickle_materializer', attribute='CloudpickleMaterializer', type=<SourceType.INTERNAL: 'internal'>), data_type=Source(module='builtins', attribute='dict', type=<SourceType.BUILTIN: 'builtin'>), tags=[]), metadata=None), ArtifactVersionResponse(id=UUID('ff1bafe8-f0b5-4727-b727-866fbca5870b'), permission_denied=False, body=ArtifactVersionResponseBody(created=datetime.datetime(2024, 1, 19, 20, 17, 8, 388199), updated=datetime.datetime(2024, 1, 19, 20, 17, 8, 388199), user=UserResponse(id=UUID('e635e34b-d62d-4921-a256-79a5db59272b'), permission_denied=False, body=UserResponseBody(created=datetime.datetime(2024, 1, 4, 13, 34, 12, 602287), updated=datetime.datetime(2024, 1, 4, 13, 34, 12, 602287), active=True, activation_token=None, full_name='', email_opted_in=None, is_service_account=False), metadata=None, name='default'), artifact=ArtifactResponse(id=UUID('df4372da-0e7e-4383-a561-f9526d4d58cb'), permission_denied=False, body=ArtifactResponseBody(created=datetime.datetime(2024, 1, 18, 13, 29, 41, 231649), updated=datetime.datetime(2024, 1, 18, 13, 29, 41, 231649), tags=[]), metadata=None, name='inference_pipeline::create_input_dataframe::output'), version='5', uri='C:\\Users\\rajar\\AppData\\Roaming\\zenml\\local_stores\\dbc029b4-f329-4ec0-88bd-c107eba71b9e\\create_input_dataframe\\output\\7bb2b729-0374-4e72-a5bb-2ca74a46a0ec', type=<ArtifactType.DATA: 'DataArtifact'>, materializer=Source(module='zenml.materializers.numpy_materializer', attribute='NumpyMaterializer', type=<SourceType.INTERNAL: 'internal'>), data_type=Source(module='numpy', attribute='ndarray', type=<SourceType.UNKNOWN: 'unknown'>), tags=[]), metadata=None), ArtifactVersionResponse(id=UUID('7f2fd697-9a46-4665-9cf1-36a94ede328b'), permission_denied=False, body=ArtifactVersionResponseBody(created=datetime.datetime(2024, 1, 19, 20, 17, 9, 152369), updated=datetime.datetime(2024, 1, 19, 20, 17, 9, 152369), user=UserResponse(id=UUID('e635e34b-d62d-4921-a256-79a5db59272b'), permission_denied=False, body=UserResponseBody(created=datetime.datetime(2024, 1, 4, 13, 34, 12, 602287), updated=datetime.datetime(2024, 1, 4, 13, 34, 12, 602287), active=True, activation_token=None, full_name='', email_opted_in=None, is_service_account=False), metadata=None, name='default'), artifact=ArtifactResponse(id=UUID('aa6bd6c1-8e56-4bf9-9daf-6aa4ce9ea40c'), permission_denied=False, body=ArtifactResponseBody(created=datetime.datetime(2024, 1, 18, 15, 35, 8, 78389), updated=datetime.datetime(2024, 1, 18, 15, 35, 8, 78389), tags=[]), metadata=None, name='inference_pipeline::predict_data::output'), version='4', uri='C:\\Users\\rajar\\AppData\\Roaming\\zenml\\local_stores\\dbc029b4-f329-4ec0-88bd-c107eba71b9e\\predict_data\\output\\bbc4e163-6e4d-4703-934c-bdd4f8a4c86a', type=<ArtifactType.DATA: 'DataArtifact'>, materializer=Source(module='zenml.materializers.built_in_materializer', attribute='BuiltInMaterializer', type=<SourceType.INTERNAL: 'internal'>), data_type=Source(module='numpy', attribute='float64', type=<SourceType.UNKNOWN: 'unknown'>), tags=[]), metadata=None)]
list = []
for i in range(0, 26, 1):
    list.append(i)
    list.append(i+0.5)
print(list)