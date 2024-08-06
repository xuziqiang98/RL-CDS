from src.envs.vertexsystem import VertexSystemFactory

def make(id, *args, **kwargs):

    if id == "VertexSystem":
        env = VertexSystemFactory.get(*args, **kwargs)

    else:
        raise NotImplementedError()

    return env