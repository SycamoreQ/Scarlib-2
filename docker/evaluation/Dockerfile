FROM azul/zulu-openjdk:17-latest
RUN mkdir /experiment
WORKDIR /experiment
COPY . ./
RUN apt update && apt install -y python3 && apt install -y python3-pip
RUN pip install -r requirements.txt
RUN ./gradlew --stop 
CMD ./gradlew simpleExperimentEval