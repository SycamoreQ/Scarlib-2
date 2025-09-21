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
    // Core Scala
    implementation("org.scala-lang:scala-library:2.13.10")

    // ScaRLib + DSL
    implementation("io.github.davidedomini:scarlib-core:$scarlibVersion")
    implementation("io.github.davidedomini:dsl-core:$scarlibVersion")
    implementation("io.github.davidedomini:alchemist-scafi:$scarlibVersion")

    // Alchemist
    implementation("it.unibo.alchemist:alchemist:25.14.6")
    implementation("it.unibo.alchemist:alchemist-incarnation-scafi:25.14.6")
    implementation("it.unibo.alchemist:alchemist-incarnation-protelis:25.14.6")
    implementation("it.unibo.alchemist:alchemist-swingui:25.7.1")

    // Python interop
    implementation("dev.scalapy:scalapy-core_2.13:0.5.3")
    implementation("ai.kien:python-native-libs_3:0.2.4")

    // Logging
    implementation("org.slf4j:slf4j-api:2.0.6")
    implementation("ch.qos.logback:logback-classic:1.4.5")

    // Spark (Scala 2.13 build)
    implementation("org.apache.spark:spark-core_2.13:3.4.0")
    implementation("org.apache.spark:spark-sql_2.13:3.4.0")

    // Force json4s to match Spark 3.4.0 expectations
    implementation("org.json4s:json4s-core_2.13:3.7.0-M11")
    implementation("org.json4s:json4s-jackson_2.13:3.7.0-M11")
}

configurations.all {
    // Avoid Spark accidentally pulling Scala 2.12 or conflicting json4s
    exclude(group = "org.json4s", module = "json4s-core_2.12")
    exclude(group = "org.json4s", module = "json4s-jackson_2.12")
    exclude(group = "org.json4s", module = "json4s-native_2.12")
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

tasks.named<JavaExec>("simpleExperimentTrainingGui") {
    jvmArgs(
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED",
        "-Dio.netty.tryReflectionSetAccessible=true"
    )
}


