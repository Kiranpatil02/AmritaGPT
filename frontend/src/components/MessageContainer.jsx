import React from "react";
import UserChat from "./UserChat";
import GPTChat from "./GPTChat";

const MessageContainer = () => {
  return (
    <div className="px-6 mb-4">
      <GPTChat message={"Hey I am Amrita-gpt, Ask any queries related to Amrita Univeristy"} />
    </div>
  );
};

export default MessageContainer;
