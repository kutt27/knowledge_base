**Objective:**
- You are able to tranform your project into a docker image.
- Run it as a docker container
- Inspect the container
- Run some telemetry

## Things you should know

Almost everything in Docker comes down to four objects. 

1. Docker image
2. Docker container
3. Docker volume
4. Docker networking

### Images

An image is a read-only template that bundles an app together with everything it needs to run. At first you rarely build one by hand; you pull it from a registry like Docker Hub. The `docker pull` command fetches an image, `docker images` lists what you have locally, and `docker rmi` removes one you no longer need:

```bash
docker pull nginx
docker images
docker rmi nginx
```

Images are built from layers, and each layer is cached. When two images share a base, Docker stores that base once and reuses it. It is also why a change near the top of a Dockerfile can rebuild almost nothing while a change near the bottom rebuilds a lot.

### Containers

A container is a running instance of an image, with a thin writable layer on top. The image itself stays untouched; anything written during the container's life goes into that disposable layer. This is the lifecycle you will type over and over:

```bash
docker run -d -p 8080:80 --name web nginx
docker ps
docker logs web
docker exec -it web sh
docker stop web
docker start web
docker rm web
```

Detached mode with `-d` runs the container in the background so your terminal stays free. `docker ps` shows what is currently running, and adding `-a` includes the stopped ones. `docker exec` drops you inside a running container, which is the fastest way to poke around when something misbehaves. And `docker rm` deletes the container for good, which is exactly why its writable layer cannot be trusted with anything important.

### Volumes

Because a container's own storage disappears with it, anything worth keeping has to live outside. A volume is storage on the host that Docker manages and mounts into a container, so the data survives even when the container is replaced. Create one, then attach it at the path where the app expects to write:

```bash
docker volume create pgdata
docker run -d --name db \
  -e POSTGRES_PASSWORD=secret \
  -v pgdata:/var/lib/postgresql/data \
  postgres
```

There is a second option called a bind mount, which maps a folder from your own machine straight into the container. Use a named volume as above for data that must persist, like a database, and use a bind mount when you want live code edits to show up instantly during development:

```bash
docker run -d -p 3000:3000 -v $(pwd):/app my-api
```

`docker volume ls` lists volumes and `docker volume rm` removes them. 

> Containers are meant to be thrown away, therefore, if anything that needs persistence, we should keep that in volume or bind it to our own vm/host device.

### Networking

Every container gets a network interface. By default it lands on Docker's bridge network, where it can reach the outside world but cannot reach other containers by name. Port mapping is how the outside reaches in:

```bash
docker run -d -p 8080:80 nginx
```

That maps host port 8080 to container port 80. The host side is what you open in the browser; the container side is what the app is actually listening on, and the two numbers do not have to match. Keep this in mind, because the practice project uses exactly that split.

When containers need to talk to each other, put them on the same user-defined network. Docker runs a small DNS service on that network, so every container answers to its own name instead of an IP address:

```bash
docker network create app-net
docker run -d --name db --network app-net postgres
docker run -d --name api --network app-net my-api
```

Now the API can connect to `db:5432` directly. This is the same trick Compose performs for you later, and it is the most common source of confusion, so it is worth doing once by hand before Compose hides it.

## Dockerfile

Every Dockerfile have the same structure how big or small it is. 

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`FROM` picks the base image everything else is built on. `WORKDIR` sets the folder inside the image where the following commands run. `COPY` brings files from your project into the image, and the first one copies only the dependency list. `RUN` executes a command while building, which is where dependencies get installed. The second `COPY` brings in the rest of your source. `EXPOSE` documents the port the app listens on, and `CMD` is the command that actually starts when the container runs.

The one subtlety is why the dependencies get their own `COPY` line before the source. Docker caches each step, so as long as `requirements.txt` has not changed, it reuses the installed-dependency layer and only redoes the steps below it. When you edit your source code, the install step is skipped entirely and rebuilds stay fast. Reorder those two lines and every code change reinstalls everything.

## Practice

For practice, we use a small FastAPI application backed by a Postgres database. Go to [github.com/marciovrl/fastapi](https://github.com/marciovrl/fastapi). Clone ig.

```bash
git clone https://github.com/marciovrl/fastapi.git
cd fastapi
docker compose up -d --build
```

Once it is running, open its interactive API documentation in the browser:

```
http://127.0.0.1:8002/docs
```

Notice the port. Compose maps host port 8002 to container port 8000, and the app itself listens on 8000, which is the exact host-versus-container split from the networking section. You reach it on 8002; the container knows nothing about that number. 

```bash
docker compose ps
docker compose logs -f app
docker compose down
```

Read the Dockerfile for this application.

```dockerfile
FROM python:3.8.1-alpine

WORKDIR /src

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /src/requirements.txt

RUN set -eux \
    && apk add --no-cache --virtual .build-deps build-base \
    libressl-dev libffi-dev gcc musl-dev python3-dev \
    postgresql-dev \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r /src/requirements.txt \
    && rm -rf /root/.cache/pip

COPY . /src/
```

The Compose file defines the two services and how they connect:

```yaml
version: "3.7"

services:
  app:
    build: .
    container_name: app
    command: uvicorn app.main:app --reload --workers 1 --host 0.0.0.0 --port 8000
    volumes:
      - ./:/src/
    ports:
      - 8002:8000
    environment:
      - DATABASE_URL=postgresql://fastapi:fastapi@db/fastapi

  db:
    image: postgres:12.1-alpine
    container_name: db
    volumes:
      - postgres_data:/var/lib/postgresql/data/
    environment:
      - POSTGRES_USER=fastapi
      - POSTGRES_PASSWORD=fastapi
      - POSTGRES_DB=fastapi

volumes:
  postgres_data:
```

There is a lot worth noticing here. The `app` service builds from the Dockerfile in the same folder, while the `db` service just pulls the official Postgres image. The `- ./:/src/` line is a bind mount, so live edits to the code appear inside the container instantly. The `DATABASE_URL` points at the hostname `db`, not `localhost`, which is the user-defined network doing its job: Compose puts both services on one network and lets the app find the database by service name. And the named volume `postgres_data` holds the database files outside the container, so the data survives a restart. Run the tests to see the two services talking:

```bash
docker compose exec app pytest test/test.py
```

## Keeping builds lean

A build sends a context to the daemon, and by default that means everything in your folder. Add a `.dockerignore` so the waste never leaves your machine:

```gitignore
__pycache__
*.pyc
.git
.env
.venv
```

This keeps dependency folders, secrets, and build output out of the image. The result is a faster, smaller, safer build, and it also stops leaking a local `.env` into an image you might share.

## When something breaks

Debugging follows one repeatable order, and it is worth saying out loud: logs first, Compose configuration second, networking and environment variables third. Concretely, check what is running with `docker compose ps`, read the failing service with `docker compose logs -f <service>`, step inside it with `docker exec -it <container> sh`, then inspect ports and environment variables. Rebuild with `docker compose up --build` only when code or the Dockerfile has actually changed, and tidy up with `docker compose down`.

## The one-page cheat sheet

```bash
docker build -t app .
docker run -d -p 3000:3000 app
docker ps
docker logs -f <container>
docker exec -it <container> sh
docker network create app-net
docker volume create pgdata
docker compose up --build
docker compose down
```

## What to leave for later

Do not spend the first class on Kubernetes, Swarm, or production CI/CD. They matter, but they pull attention away from the core loop of build, run, and compose. A natural end goal is a trainee who has run a real multi-service project, can read its Dockerfile and Compose file without guessing, and can start or stop the whole thing with one Compose command.
