plugins {
    id("java")
    scala
}

group = "io.github.davidedomini"
version = "1.0.0"

repositories {
    mavenCentral()
}

scala {
    zincVersion.set("1.6.1")
}


sourceSets {
    main {
        scala {
            setSrcDirs(listOf("src/main/scala"))
        }
    }
    test {
        scala {
            setSrcDirs(listOf("src/test/scala"))
        }
    }
}

val scarlibVersion = "3.1.2"

dependencies {
       implementation("org.scala-lang:scala-library:2.13.10")
    implementation("io.github.davidedomini:scarlib-core:$scarlibVersion")
    implementation("io.github.davidedomini:dsl-core:$scarlibVersion")
    implementation("io.github.davidedomini:alchemist-scafi:$scarlibVersion")
    implementation("it.unibo.alchemist:alchemist:25.14.6")
    implementation("it.unibo.alchemist:alchemist-incarnation-scafi:25.14.6")
    implementation("it.unibo.alchemist:alchemist-incarnation-protelis:25.14.6")
    implementation("it.unibo.alchemist:alchemist-swingui:25.7.1")
    implementation("dev.scalapy:scalapy-core_2.13:0.5.3")
    // https://mvnrepository.com/artifact/ai.kien/python-native-libs
    implementation("ai.kien:python-native-libs_3:0.2.4")
    implementation("org.slf4j:slf4j-api:2.0.6")
    implementation("ch.qos.logback:logback-classic:1.4.5")
    implementation("org.apache.spark:spark-core_2.13:3.4.0")
    implementation("org.apache.spark:spark-sql_2.13:3.4.0")
}


tasks.register<JavaExec>("simpleExperimentTraining") {
    group = "ScaRLib Training"
    mainClass.set("experiments.training.SimpleExperimentTraining")
    jvmArgs("-Dsun.java2d.opengl=false")
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.register<JavaExec>("simpleExperimentTrainingGui") {
    group = "ScaRLib Training"
    mainClass.set("experiments.training.SimpleExperimentTraining")
    jvmArgs("-Dsun.java2d.opengl=false")
    args = listOf("20")
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.register<JavaExec>("simpleExperimentEval") {
    group = "ScaRLib Training"
    mainClass.set("experiments.evaluation.SimpleExperimentEval")
    jvmArgs("-Dsun.java2d.opengl=false")
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.register<JavaExec>("simpleExperimentEvalGui") {
    group = "ScaRLib Training"
    mainClass.set("experiments.evaluation.SimpleExperimentEval")
    jvmArgs("-Dsun.java2d.opengl=false")
    args = listOf("20")
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.named<JavaExec>("simpleExperimentTrainingGui") {
    group = "ScaRLib Training"
    mainClass.set("vmas.MainEpidemic")
    classpath = sourceSets.main.get().runtimeClasspath
    environment("SCALAPY_PYTHON_LIBRARY", "/opt/homebrew/opt/python@3.9/Frameworks/Python.framework/Versions/3.9/lib/libpython3.9.dylib")
}


