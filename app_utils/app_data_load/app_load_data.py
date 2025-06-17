from aind_analysis_arch_result_access.han_pipeline import get_session_table


class AppLoadData:

    def __init__(self):
        self.session_table = None
        self.load()

    def load(self, load_bpod=False):
        """
        Loads session dataframe

        Args:
            load_bpod (bool): Whether to load bpod data

        Returns:
            pd.DataFrame: Loaded session table
        """
        try:
            self.session_table = get_session_table(if_load_bpod=load_bpod)
            return self.session_table
        except Exception as e:
            raise ValueError(f"Failed to load session table: {str(e)}")

    def get_data(self):
        """
        Returns current session table

        Returns:
            pd.DataFrame: Session table
        """
        if self.session_table is None:
            self.load()
        return self.session_table
