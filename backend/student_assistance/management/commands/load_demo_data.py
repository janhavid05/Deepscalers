from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from student_assistance.models import Question, Answer, KnowledgeBase
from student_assistance.ai_service import AIService

User = get_user_model()

class Command(BaseCommand):
    help = 'Loads demo data for the student assistance system'

    def handle(self, *args, **kwargs):
        # Create demo users if they don't exist
        student, _ = User.objects.get_or_create(
            email='student@example.com',
            defaults={
                'is_staff': False,
                'is_active': True,
                'first_name': 'Demo',
                'last_name': 'Student',
                'phone_number': '+1234567890',
                'is_verified': True
            }
        )
        student.set_password('demo123')
        student.save()

        faculty, _ = User.objects.get_or_create(
            email='faculty@example.com',
            defaults={
                'is_staff': True,
                'is_active': True,
                'first_name': 'Demo',
                'last_name': 'Faculty',
                'phone_number': '+1234567891',
                'is_verified': True
            }
        )
        faculty.set_password('demo123')
        faculty.save()

        # Sample Q&A pairs
        demo_data = [
            {
                'question': 'What is the difference between supervised and unsupervised learning?',
                'answer': 'Supervised learning uses labeled data to train models, where each example has a known output. Unsupervised learning works with unlabeled data, finding patterns and structures without predefined outputs. Supervised learning is used for classification and regression tasks, while unsupervised learning is used for clustering and dimensionality reduction.'
            },
            {
                'question': 'How does gradient descent work in machine learning?',
                'answer': 'Gradient descent is an optimization algorithm that minimizes a function by iteratively moving in the direction of steepest descent. In machine learning, it\'s used to find the optimal parameters of a model by minimizing the loss function. The algorithm calculates the gradient of the loss function with respect to the parameters and updates them in the opposite direction of the gradient.'
            },
            {
                'question': 'What is overfitting in machine learning?',
                'answer': 'Overfitting occurs when a model learns the training data too well, including its noise and outliers, resulting in poor generalization to new data. Signs of overfitting include high accuracy on training data but low accuracy on test data. It can be prevented using techniques like regularization, cross-validation, and early stopping.'
            },
            {
                'question': 'Explain the concept of cross-validation.',
                'answer': 'Cross-validation is a technique to assess how well a model generalizes to an independent dataset. It involves splitting the data into k subsets (folds), training the model k times using k-1 folds for training and the remaining fold for validation. This helps in getting a more robust estimate of model performance and detecting overfitting.'
            },
            {
                'question': 'What are the main types of neural network architectures?',
                'answer': 'The main types of neural network architectures include: 1) Feedforward Neural Networks (FNN) - basic architecture with forward connections, 2) Convolutional Neural Networks (CNN) - specialized for image processing, 3) Recurrent Neural Networks (RNN) - designed for sequential data, 4) Long Short-Term Memory (LSTM) - a type of RNN for long-term dependencies, 5) Transformer - modern architecture using self-attention mechanisms.'
            }
        ]

        # Initialize AI service
        ai_service = AIService()

        # Create questions, answers, and knowledge base entries
        for item in demo_data:
            # Create question
            question = Question.objects.create(
                student=student,
                text=item['question'],
                status='answered'
            )

            # Create answer
            answer = Answer.objects.create(
                question=question,
                faculty=faculty,
                text=item['answer'],
                is_verified=True
            )

            # Add to knowledge base
            KnowledgeBase.objects.create(
                question=item['question'],
                answer=item['answer'],
                vector_embedding=ai_service.get_embedding(f"{item['question']}\n{item['answer']}"),
                confidence_score=1.0
            )

            # Add to Qdrant
            ai_service.add_to_knowledge_base(
                question=item['question'],
                answer=item['answer'],
                confidence_score=1.0
            )

        self.stdout.write(self.style.SUCCESS('Successfully loaded demo data')) 