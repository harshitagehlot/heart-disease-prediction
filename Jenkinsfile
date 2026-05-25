pipeline {
    agent any

    stages {

        stage('Clone Repo') {
            steps {
                git branch: 'main', url: 'https://github.com/harshitagehlot/heart-disease-prediction'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t heart-disease-app .'
            }
        }

        stage('Run Container') {
            steps {
                bat 'docker run -d -p 5000:5000 heart-disease-app'
            }
        }
    }
}