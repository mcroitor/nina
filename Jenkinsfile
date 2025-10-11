pipeline {
    agent any

    parameters {
        string(
            name: 'BRANCH_NAME',
            defaultValue: 'main',
            description: 'Branch to build'
        )
    }

    environment {
        PHP_VERSION = '8.1'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout([
                    $class: 'GitSCM', 
                    branches: [[name: "*/${params.BRANCH_NAME}"]], 
                    userRemoteConfigs: scm.userRemoteConfigs
                ])
            }
        }
        stage('Test') {
            steps {
                dir('tests') {
                    sh 'php test_cell.php'
                    sh 'php test_layer.php'
                    sh 'php test_exceptions.php'
                }
            }
        }
        stage('Train') {
            steps {
                dir('examples/letter_recognition') {
                    sh 'php letter_recognition.php'
                }
            }
        }
    }
    post {
        always {
            echo 'This will always run after the stages.'
        }
        success {
            echo 'This will run only if the pipeline succeeds.'
        }
        failure {
            echo 'This will run only if the pipeline fails.'
        }
    }
}