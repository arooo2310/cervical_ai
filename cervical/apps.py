from django.apps import AppConfig
import threading
import os
import sys

class CervicalConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'cervical'

    def ready(self):
        # Prevent running twice with auto-reloader
        # In runserver, 'runserver' command starts, then spawns child with RUN_MAIN=true.
        # We generally want to run in the child process.
        if os.environ.get('RUN_MAIN', None) != 'true':
            # This is likely the reloader process (or simple run without reload)
            # If checking strict RUN_MAIN, we might miss --noreload case, but for dev it's fine.
            # To be safer: strict check for RUN_MAIN ensures we don't start in reloader.
            pass
        
        # Check if we are running 'runserver'. If we are running migration/other commands, we shouldn't start FL.
        # sys.argv check is a heuristic.
        is_runserver = any('runserver' in arg for arg in sys.argv)
        
        if is_runserver and os.environ.get('RUN_MAIN') == 'true':
            import subprocess
            import atexit
            from django.conf import settings
            
            print(" >> Starting Federated Learning Server in a subprocess...")
            
            # python fed_server.py
            base_dir = settings.BASE_DIR
            script_path = os.path.join(base_dir, 'federated', 'fed_server.py')
            python_exe = sys.executable
            
            try:
                # Start process
                proc = subprocess.Popen([python_exe, script_path])
                
                # Ensure cleanup on exit
                def cleanup_fl_server():
                    if proc.poll() is None:
                        print(" >> Terminating FL server subprocess...")
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            
                atexit.register(cleanup_fl_server)
                
            except Exception as e:
                print(f" >> FAILED to start FL server subprocess: {e}")
