"""
Main Module - Entry point for AIML Lab demonstrations
This module imports and runs demonstrations from numpy_module and pandas_module
"""

import sys


def display_menu():
    """Display the main menu"""
    print("\n" + "=" * 60)
    print(" " * 15 + "AIML LAB - MODULE DEMONSTRATIONS")
    print("=" * 60)
    print("\nAvailable Modules:")
    print("  1. NumPy Module - Array Operations and Linear Algebra")
    print("  2. Pandas Module - Data Analysis and Manipulation")
    print("  3. Run All Demonstrations")
    print("  0. Exit")
    print("=" * 60)


def run_numpy_module():
    """Import and run NumPy module demonstrations"""
    try:
        import numpy_module
        print("\n" + "🔢 " * 20)
        print("RUNNING NUMPY MODULE")
        print("🔢 " * 20)
        # The module will run automatically when imported due to __main__ block
    except ImportError as e:
        print(f"Error importing numpy_module: {e}")
        print("Make sure numpy_module.py is in the same directory.")
    except Exception as e:
        print(f"Error running NumPy demonstrations: {e}")


def run_pandas_module():
    """Import and run Pandas module demonstrations"""
    try:
        import pandas_module
        print("\n" + "🐼 " * 20)
        print("RUNNING PANDAS MODULE")
        print("🐼 " * 20)
        pandas_module.main()
    except ImportError as e:
        print(f"Error importing pandas_module: {e}")
        print("Make sure pandas_module.py is in the same directory.")
    except Exception as e:
        print(f"Error running Pandas demonstrations: {e}")


def main():
    """Main function to run the program"""
    while True:
        display_menu()
        
        try:
            choice = input("\nEnter your choice (0-3): ").strip()
            
            if choice == '1':
                run_numpy_module()
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                run_pandas_module()
                input("\nPress Enter to continue...")
                
            elif choice == '3':
                run_numpy_module()
                print("\n" + "-" * 60 + "\n")
                run_pandas_module()
                input("\nPress Enter to continue...")
                
            elif choice == '0':
                print("\n👋 Thank you for using AIML Lab demonstrations!")
                print("Goodbye!\n")
                sys.exit(0)
                
            else:
                print("\n❌ Invalid choice! Please enter a number between 0 and 3.")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted. Goodbye!\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    print("\n" + "🚀 " * 20)
    print(" " * 15 + "Welcome to AIML Lab!")
    print("🚀 " * 20)
    main()
