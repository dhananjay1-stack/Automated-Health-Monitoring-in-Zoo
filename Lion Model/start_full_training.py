
from training import main
import logging
import time

# Configure logging for detailed monitorring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('training_log.txt')  # File outpt for later review
    ]
)

logger = logging.getLogger(__name__)

def start_complete_training():
    """
    Start the complete Lion Behavior Detection training pipeline
    This will run all phases: training, validation, evaluation, and deployment
    """
    
    logger.info("🦁" + "="*70)
    logger.info("   HYBRID LION BEHAVIOR DETECTION SYSTEM - FULL TRAINING")
    logger.info("="*70)
    logger.info("🎯 MISSION: Detect abnormal behavior/injuries in lions")
    logger.info("🏥 PURPOSE: Enable timely veterinary intervention")
    logger.info("🏗️ ARCHITECTURE: TimeSformer + PoseTransformer + Advanced Fusion")
    logger.info("="*70)
    
    # Display training configuration
    logger.info("\n📋 TRAINING CONFIGURATION:")
    logger.info("   • Dataset: Your lion video dataset with pose keypoints")
    logger.info("   • Model: 11.2M parameter hybrid architecture")
    logger.info("   • Training: Mixed precision with multi-loss optimization")
    logger.info("   • Validation: Medical-grade metrics (sensitivity/specificity)")
    logger.info("   • Output: Production-ready veterinary alert system")
    
    # Start training
    logger.info("\n🚀 INITIATING TRAINING PIPELINE...")
    logger.info("   This may take 30-60 minutes depending on your hardware")
    logger.info("   Progress will be logged in real-time below")
    logger.info("-" * 70)
    
    start_time = time.time()
    
    try:
        # Call the main training function
        results = main()
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        if results:
            logger.info("\n" + "🎉" * 20)
            logger.info("   TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("🎉" * 20)
            logger.info(f"⏱️  Total Training Time: {training_duration/60:.1f} minutes")
            logger.info(f"📈 Model Performance: {results.get('metrics', {}).get('accuracy', 'N/A')}")
            logger.info(f"📦 Model Location: {results.get('export_path', 'checkpoints/')}")
            logger.info("\n🏥 VETERINARY SYSTEM READY FOR DEPLOYMENT!")
            
            return results
            
        else:
            logger.error("❌ Training pipeline failed - check logs above")
            return None
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        logger.info("💾 Partial progress saved in checkpoints/")
        return None
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def monitor_training_progress():
    """
    Display what to expect during training
    """
    logger.info("\n👀 WHAT TO EXPECT DURING TRAINING:")
    logger.info("="*50)
    
    phases = [
        "📂 Data Loading & Validation",
        "🏗️  Model Architecture Initialization", 
        "⚙️  Training Loop with Mixed Precision",
        "📊 Validation Metrics Computation",
        "💾 Checkpoint Saving (Best Models)",
        "🔬 Comprehensive Model Evaluation",
        "🚀 Production Deployment Preparation",
        "🧪 Inference Pipeline Testing",
        "🏥 Veterinary Recommendation System Setup"
    ]
    
    for i, phase in enumerate(phases, 1):
        logger.info(f"   {i}. {phase}")
    
    logger.info("\n📈 METRICS YOU'LL SEE:")
    logger.info("   • Training/Validation Loss & Accuracy")
    logger.info("   • F1-Score (Weighted for class balance)")
    logger.info("   • Sensitivity (Abnormal behavior detection rate)")
    logger.info("   • Specificity (False alarm reduction)")
    logger.info("   • AUC Score (Overall discrimination ability)")
    
    logger.info("\n💾 FILES THAT WILL BE CREATED:")
    logger.info("   • checkpoints/best_model.pth - Best performing model")
    logger.info("   • evaluation_results/ - Detailed performance analysis")
    logger.info("   • deployed_model/ - Production-ready model export")
    logger.info("   • training_results.json - Complete training summary")
    logger.info("   • training_log.txt - Detailed training logs")

if __name__ == "__main__":
    print("🦁 Lion Behavior Detection - Full Training Pipeline")
    print("="*55)
    
    # Show what to expect
    monitor_training_progress()
    
    # Ask for confirmation
    print("\n" + "⚠️ " * 15)
    print("READY TO START COMPLETE TRAINING PIPELINE?")
    print("This will train the model on your full dataset")
    print("⚠️ " * 15)
    
    confirm = input("\nProceed with training? (y/N): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        print("\n🚀 Starting training pipeline...")
        results = start_complete_training()
        
        if results:
            print("\n✅ SUCCESS! Your lion behavior detection system is ready!")
            print("🏥 Deploy in veterinary facilities for real-time monitoring")
        else:
            print("\n❌ Training encountered issues - check logs for details")
    else:
        print("\n👋 Training cancelled - run again when ready!")